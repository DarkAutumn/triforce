import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

interface MilestoneEvent {
  schema_version: number;
  type: "milestone";
  id: number;
  timestamp: number;
  reason: string;
  run_dir: string;
  experiment_dir: string;
  step: number;
  scenario: string | null;
  report_path: string;
  checkpoint_path: string | null;
  anomalies: unknown[];
}

interface CurrentStatus {
  name?: string | null;
  steps?: number;
  total_steps?: number;
  pct?: number;
}

interface OverallStatus {
  steps?: number;
  total_steps?: number;
  pct?: number;
  sps?: number | null;
  eta_seconds?: number | null;
}

interface TriforceStatus {
  state?: string;
  updated_at?: number;
  pid?: number;
  experiment_dir?: string;
  run_dir?: string;
  scenario?: string;
  current?: CurrentStatus;
  overall?: OverallStatus;
  latest_metrics?: Record<string, unknown>;
  latest_stats?: Record<string, unknown>;
  latest_checkpoint_path?: string | null;
  last_milestone_step?: number | null;
  last_milestone_reason?: string | null;
}

interface ProcessOutput {
  stdoutPath: string;
  stderrPath: string;
  stdoutTail: string[];
  stderrTail: string[];
}

interface ExperimentState {
  experimentId: string;
  experimentDir: string;
  outputRoot: string;
  runDir: string | null;
  childPid: number | null;
  child: ChildProcess | null;
  eventsOffset: number;
  pendingMilestones: MilestoneEvent[];
  wakeInFlight: boolean;
  widgetTimer: NodeJS.Timeout | null;
  watchTimer: NodeJS.Timeout | null;
  guardrailTimer: NodeJS.Timeout | null;
  status: TriforceStatus | null;
  processOutput: ProcessOutput | null;
  failureWakeSent: boolean;
}

interface StartParams {
  experiment_id?: string;
  scenario: string;
  action_space: string;
  model_kind: string;
  iterations?: number;
  parallel: number;
  load?: string;
  resume: boolean;
  skip_to?: string;
  baseline_eval_json?: string;
  extra_args: string[];
}

interface RestartParams {
  checkpoint_path: string;
  scenario: string;
  action_space: string;
  model_kind: string;
  skip_to?: string;
  iterations?: number;
  parallel: number;
  baseline_eval_json?: string;
  extra_args: string[];
}

interface UiHooks {
  hasUI: () => boolean;
  isIdle: () => boolean;
  setWidget: (lines: string[]) => void | Promise<void>;
  notify: (message: string, level: "info" | "warn" | "error") => void | Promise<void>;
}

let state: ExperimentState | null = null;
let uiHooks: UiHooks | null = null;
let wakeToolTouched = false;

export default function triforceExperiment(pi: ExtensionAPI): void {
  const z = pi.zod;
  pi.setLabel("Triforce Experiments");

  pi.on("session_start", async (_event, ctx) => {
    uiHooks = {
      hasUI: () => ctx.hasUI,
      isIdle: () => ctx.isIdle(),
      setWidget: (lines: string[]) => ctx.ui.setWidget("aboveEditor", lines),
      notify: (message: string, level: "info" | "warn" | "error") => ctx.ui.notify(message, level),
    };
    reattachNewestExperiment();
    if (state) {
      startTimers(pi);
    }
  });

  pi.on("turn_end", async () => {
    if (wakeInProgress() && !wakeToolTouched) {
      if (state) {
        state.wakeInFlight = false;
      }
    }
    wakeToolTouched = false;
    await deliverNewestQueuedMilestone(pi);
  });

  pi.on("session_shutdown", async () => {
    if (state?.child && state.childPid) {
      writeControl(state, "pause", "OMP session shutdown paused attached training process");
    }
  });

  pi.registerTool({
    name: "triforce_experiment_start",
    label: "Start Triforce Experiment",
    description: "Start a headless Triforce training experiment controlled by OMP milestone wakes.",
    parameters: z.object({
      experiment_id: z.string().optional(),
      scenario: z.string(),
      action_space: z.string().default("all-items"),
      model_kind: z.string().default("impala-multihead"),
      iterations: z.number().int().positive().optional(),
      parallel: z.number().int().positive().default(16),
      load: z.string().optional(),
      resume: z.boolean().default(false),
      skip_to: z.string().optional(),
      baseline_eval_json: z.string().optional(),
      extra_args: z.array(z.string()).default([]),
    }),
    async execute(_toolCallId, params) {
      const parsed = params as StartParams;
      const result = startExperiment(pi, parsed, null);
      return toolResult(result.message, result.details, !result.ok);
    },
  });

  pi.registerTool({
    name: "triforce_experiment_status",
    label: "Triforce Experiment Status",
    description: "Return the current Triforce experiment extension state and latest status.json.",
    parameters: z.object({}),
    async execute() {
      const status = readCurrentStatus();
      return toolResult("Triforce experiment status loaded.", {
        active: state !== null,
        experiment_id: state?.experimentId ?? null,
        experiment_dir: state?.experimentDir ?? null,
        run_dir: state?.runDir ?? null,
        child_pid: state?.childPid ?? null,
        status: status ?? { state: "starting" },
      }, false);
    },
  });

  pi.registerTool({
    name: "triforce_experiment_control",
    label: "Control Triforce Experiment",
    description: "Write continue, pause, or stop to the active training run control.json.",
    parameters: z.object({
      command: z.enum(["continue", "pause", "stop"]),
      reason: z.string().default(""),
    }),
    async execute(_toolCallId, params) {
      wakeToolTouched = true;
      if (!state?.runDir) {
        return toolResult("No active run directory is known.", {}, true);
      }
      const command = readStringField(params, "command") ?? "continue";
      const reason = readStringField(params, "reason") ?? "";
      writeControl(state, command, reason);
      state.wakeInFlight = false;
      return toolResult(`Wrote ${command} to control.json.`, {
        path: path.join(state.runDir, "control.json"),
      }, false);
    },
  });

  pi.registerTool({
    name: "triforce_experiment_restart",
    label: "Restart Triforce Experiment",
    description: "Restart the current experiment from a checkpoint, preserving the experiment journal.",
    parameters: z.object({
      checkpoint_path: z.string(),
      scenario: z.string(),
      action_space: z.string().default("all-items"),
      model_kind: z.string().default("impala-multihead"),
      skip_to: z.string().optional(),
      iterations: z.number().int().positive().optional(),
      parallel: z.number().int().positive().default(16),
      baseline_eval_json: z.string().optional(),
      extra_args: z.array(z.string()).default([]),
    }),
    async execute(_toolCallId, params) {
      wakeToolTouched = true;
      if (!state) {
        return toolResult("No active experiment exists to restart.", {}, true);
      }
      const parsed = params as RestartParams;
      await stopChildForRestart(state);
      const startParams: StartParams = {
        experiment_id: state.experimentId,
        scenario: parsed.scenario,
        action_space: parsed.action_space,
        model_kind: parsed.model_kind,
        iterations: parsed.iterations,
        parallel: parsed.parallel,
        load: parsed.checkpoint_path,
        resume: parsed.skip_to === undefined,
        skip_to: parsed.skip_to,
        baseline_eval_json: parsed.baseline_eval_json,
        extra_args: parsed.extra_args,
      };
      const result = startExperiment(pi, startParams, state.experimentDir);
      if (state) {
        state.wakeInFlight = false;
      }
      return toolResult(result.message, result.details, !result.ok);
    },
  });

  pi.registerTool({
    name: "triforce_experiment_finish",
    label: "Finish Triforce Experiment",
    description: "Finish extension supervision for the current Triforce experiment.",
    parameters: z.object({
      summary_path: z.string().optional(),
      stop_training: z.boolean().default(false),
    }),
    async execute(_toolCallId, params) {
      wakeToolTouched = true;
      const summaryPath = readStringField(params, "summary_path");
      const stopTraining = readBooleanField(params, "stop_training") ?? false;
      if (state && stopTraining) {
        writeControl(state, "stop", "experiment finished by agent");
      }
      const finalStatus = readCurrentStatus();
      clearTimers();
      clearWidget();
      const finishedExperiment = state?.experimentId ?? null;
      state = null;
      return toolResult("Triforce experiment supervision finished.", {
        experiment_id: finishedExperiment,
        summary_path: summaryPath ?? null,
        status: finalStatus,
      }, false);
    },
  });
}

function startExperiment(pi: ExtensionAPI, params: StartParams, existingExperimentDir: string | null) {
  if (!isTriforceRoot(process.cwd())) {
    return { ok: false, message: "Run OMP from the Triforce repository root.", details: {} };
  }
  const pythonPath = path.join(process.cwd(), ".venv", "bin", "python");
  if (!fs.existsSync(pythonPath)) {
    return {
      ok: false,
      message: "Missing .venv/bin/python. Create the Python 3.12 venv per .github/copilot-instructions.md.",
      details: { python_path: pythonPath },
    };
  }

  clearTimers();
  const experimentId = sanitizeExperimentId(params.experiment_id ?? `${params.scenario}-${timestampSlug()}`);
  const experimentDir = existingExperimentDir ?? path.join("training", "experiments", experimentId);
  const outputRoot = path.join(experimentDir, "runs");
  fs.mkdirSync(outputRoot, { recursive: true });
  ensureJournal(experimentDir, params);

  const args = [
    "train.py",
    params.scenario,
    params.action_space,
    params.model_kind,
    "--headless-agent",
    "--experiment-dir",
    experimentDir,
    "--output",
    outputRoot,
    "--parallel",
    String(params.parallel),
  ];
  if (params.iterations !== undefined) {
    args.push("--iterations", String(params.iterations));
  }
  if (params.load) {
    args.push("--load", params.load);
  }
  if (params.resume) {
    args.push("--resume");
  }
  if (params.skip_to) {
    args.push("--skip-to", params.skip_to);
  }
  if (params.baseline_eval_json) {
    args.push("--baseline-eval-json", params.baseline_eval_json);
  }
  args.push(...params.extra_args);

  const child = spawn(pythonPath, args, {
    cwd: process.cwd(),
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
  });

  state = {
    experimentId,
    experimentDir,
    outputRoot,
    runDir: null,
    childPid: child.pid ?? null,
    child,
    eventsOffset: 0,
    pendingMilestones: [],
    wakeInFlight: false,
    widgetTimer: null,
    watchTimer: null,
    guardrailTimer: null,
    status: null,
    processOutput: makeProcessOutput(experimentDir, child.pid ?? 0),
    failureWakeSent: false,
  };

  child.stdout?.on("data", (chunk: Buffer) => {
    const output = chunk.toString("utf8");
    appendProcessOutput("stdout", output);
    const runDir = parseOutputRunDir(output);
    if (runDir && state) {
      state.runDir = runDir;
      startTimers(pi);
    }
  });
  child.stderr?.on("data", (chunk: Buffer) => {
    const text = chunk.toString("utf8");
    appendProcessOutput("stderr", text);
    const trimmed = text.trim();
    if (trimmed) {
      void uiHooks?.notify(`Triforce train.py: ${trimmed}`, "warn");
    }
  });
  child.on("exit", (code, signal) => {
    if (state?.child === child) {
      state.child = null;
    }
    if (code !== 0 || signal) {
      void sendProcessFailureWake(pi, "process_exit", `train.py exited with code=${code ?? "null"} signal=${signal ?? "null"}`);
    }
  });
  child.unref();
  startTimers(pi);

  return {
    ok: true,
    message: "Triforce experiment started.",
    details: {
      experiment_id: experimentId,
      experiment_dir: experimentDir,
      run_dir: state.runDir,
      child_pid: state.childPid,
      journal_path: path.join(experimentDir, "journal.md"),
      output_root: outputRoot,
    },
  };
}

function startTimers(pi: ExtensionAPI): void {
  if (!state) {
    return;
  }
  if (!state.widgetTimer) {
    state.widgetTimer = setInterval(updateWidget, 30_000);
    updateWidget();
  }
  if (!state.watchTimer) {
    state.watchTimer = setInterval(() => pollEvents(pi), 1_000);
  }
  if (!state.guardrailTimer) {
    state.guardrailTimer = setInterval(() => enforceExtensionGuardrail(pi), 60_000);
  }
}

function clearTimers(): void {
  if (!state) {
    return;
  }
  if (state.widgetTimer) {
    clearInterval(state.widgetTimer);
  }
  if (state.watchTimer) {
    clearInterval(state.watchTimer);
  }
  if (state.guardrailTimer) {
    clearInterval(state.guardrailTimer);
  }
  state.widgetTimer = null;
  state.watchTimer = null;
  state.guardrailTimer = null;
}

function pollEvents(pi: ExtensionAPI): void {
  if (!state?.runDir) {
    return;
  }
  const eventsPath = path.join(state.runDir, "events.jsonl");
  if (!fs.existsSync(eventsPath)) {
    return;
  }
  const text = fs.readFileSync(eventsPath, "utf8");
  const lastNewline = text.lastIndexOf("\n");
  if (lastNewline < state.eventsOffset) {
    return;
  }
  const chunk = text.slice(state.eventsOffset, lastNewline + 1);
  state.eventsOffset = lastNewline + 1;
  for (const line of chunk.split("\n")) {
    if (!line.trim()) {
      continue;
    }
    const parsed = parseJsonObject(line);
    if (isMilestoneEvent(parsed)) {
      queueOrDeliverMilestone(pi, parsed);
    }
  }
}

function queueOrDeliverMilestone(pi: ExtensionAPI, milestone: MilestoneEvent): void {
  if (!state) {
    return;
  }
  if (!state.wakeInFlight && (uiHooks?.isIdle() ?? true)) {
    void sendMilestoneWake(pi, milestone, 0);
    return;
  }
  state.pendingMilestones.push(milestone);
}

async function deliverNewestQueuedMilestone(pi: ExtensionAPI): Promise<void> {
  if (!state || state.wakeInFlight || state.pendingMilestones.length === 0) {
    return;
  }
  if (!(uiHooks?.isIdle() ?? true)) {
    return;
  }
  const skipped = Math.max(0, state.pendingMilestones.length - 1);
  const milestone = state.pendingMilestones[state.pendingMilestones.length - 1];
  state.pendingMilestones = [];
  await sendMilestoneWake(pi, milestone, skipped);
}

async function sendMilestoneWake(pi: ExtensionAPI, milestone: MilestoneEvent, skipped: number): Promise<void> {
  if (!state) {
    return;
  }
  const report = readTextIfExists(milestone.report_path) ?? `Report missing: ${milestone.report_path}`;
  const superseded = skipped > 0 ? `\n\n${skipped} older milestone(s) were superseded by this newest milestone.` : "";
  const message = [
    `Triforce training milestone: ${milestone.reason}`,
    superseded,
    report,
    "",
    `Journal: ${path.join(state.experimentDir, "journal.md")}`,
    `Status: ${state.runDir ? path.join(state.runDir, "status.json") : "unknown"}`,
    `Tuning: ${state.runDir ? path.join(state.runDir, "tuning.json") : "unknown"}`,
    `Control: ${state.runDir ? path.join(state.runDir, "control.json") : "unknown"}`,
    "",
    "Use the triforce-experiment skill decision loop. Decide exactly one run action: continue, stop/edit/restart, or finish. Also decide whether each wake-causing metric should keep waking, be loosened, or be disabled; cite evidence and edit tuning.json before continuing when appropriate. If continuing, call triforce_experiment_control with command continue and then yield.",
  ].join("\n");
  state.wakeInFlight = true;
  wakeToolTouched = false;
  await pi.sendMessage({
    customType: "triforce-training-milestone",
    content: message,
    display: true,
    attribution: "user",
  }, { deliverAs: "nextTurn", triggerTurn: true });
}

function updateWidget(): void {
  const status = readCurrentStatus();
  if (!state || !status || !(uiHooks?.hasUI() ?? false)) {
    return;
  }
  state.status = status;
  const current = status.current ?? {};
  const overall = status.overall ?? {};
  const metrics = status.latest_metrics ?? {};
  const legEtaSeconds = calculateLegEta(current, overall);
  const stats = status.latest_stats ?? {};
  const lines = [
    `Triforce: ${status.state ?? "unknown"} ${status.scenario ?? "unknown"} pid=${status.pid ?? state.childPid ?? "n/a"}`,
    `Run: ${state.experimentId}`,
    `Leg: ${current.name ?? "none"} ${formatInt(current.steps)}/${formatInt(current.total_steps)} ${formatPct(current.pct)} ETA=${formatEta(legEtaSeconds)}`,
    `Total: ${formatInt(overall.steps)}/${formatInt(overall.total_steps)} ${formatPct(overall.pct)} SPS=${formatNumber(overall.sps)} ETA=${formatEta(overall.eta_seconds)}`,
    `Latest: success=${formatUnknown(metrics["success-rate"])} reward=${formatUnknown(metrics["reward-average"])} entropy=${formatUnknown(stats["losses/entropy"])}`,
    `KL=${formatUnknown(stats["losses/approx_kl"])} clip=${formatUnknown(stats["losses/clipfrac"])} EV=${formatUnknown(stats["losses/explained_variance"])}`,
    `Checkpoint: ${status.latest_checkpoint_path ? path.basename(status.latest_checkpoint_path) : "none"}`,
    `Last wake: ${status.last_milestone_reason ?? "none"} @ ${status.last_milestone_step ?? "n/a"}`,
  ];
  void uiHooks?.setWidget(lines);
}

function clearWidget(): void {
  if (uiHooks?.hasUI()) {
    void uiHooks.setWidget([]);
  }
}

function readCurrentStatus(): TriforceStatus | null {
  if (!state?.runDir) {
    return state?.status ?? null;
  }
  const statusPath = path.join(state.runDir, "status.json");
  const parsed = readJsonFile(statusPath);
  if (isStatus(parsed)) {
    state.status = parsed;
    return parsed;
  }
  return state.status;
}

function reattachNewestExperiment(): void {
  const root = path.join("training", "experiments");
  if (!fs.existsSync(root)) {
    return;
  }
  const statuses = findStatusFiles(root);
  let newest: { mtime: number; status: TriforceStatus; statusPath: string } | null = null;
  for (const statusPath of statuses) {
    const parsed = readJsonFile(statusPath);
    if (!isStatus(parsed) || (parsed.state !== "running" && parsed.state !== "paused")) {
      continue;
    }
    if (typeof parsed.pid !== "number" || !isProcessAlive(parsed.pid)) {
      continue;
    }
    const stat = fs.statSync(statusPath);
    if (!newest || stat.mtimeMs > newest.mtime) {
      newest = { mtime: stat.mtimeMs, status: parsed, statusPath };
    }
  }
  if (!newest || !newest.status.experiment_dir || !newest.status.run_dir) {
    return;
  }
  const experimentId = path.basename(newest.status.experiment_dir);
  state = {
    experimentId,
    experimentDir: newest.status.experiment_dir,
    outputRoot: path.join(newest.status.experiment_dir, "runs"),
    runDir: newest.status.run_dir,
    childPid: newest.status.pid ?? null,
    child: null,
    eventsOffset: fileSize(path.join(newest.status.run_dir, "events.jsonl")),
    pendingMilestones: [],
    wakeInFlight: false,
    widgetTimer: null,
    watchTimer: null,
    guardrailTimer: null,
    status: newest.status,
    processOutput: null,
    failureWakeSent: false,
  };
}

function findStatusFiles(root: string): string[] {
  const result: string[] = [];
  for (const experimentName of safeReadDir(root)) {
    const runsRoot = path.join(root, experimentName, "runs");
    for (const scenarioName of safeReadDir(runsRoot)) {
      for (const runName of safeReadDir(path.join(runsRoot, scenarioName))) {
        const statusPath = path.join(runsRoot, scenarioName, runName, "status.json");
        if (fs.existsSync(statusPath)) {
          result.push(statusPath);
        }
      }
    }
  }
  return result;
}

function enforceExtensionGuardrail(pi: ExtensionAPI): void {
  if (!state?.runDir) {
    return;
  }
  const status = readCurrentStatus();
  const elapsed = status?.overall?.elapsed_seconds;
  if (typeof elapsed === "number" && elapsed > 604_800) {
    writeControl(state, "stop", "extension 7 day failsafe exceeded");
  }
  const updatedAt = readUpdatedAt(status);
  if (updatedAt === null) {
    return;
  }
  const staleSeconds = Date.now() / 1000 - updatedAt;
  if (staleSeconds > 600 && state.childPid && !isProcessAlive(state.childPid)) {
    void sendProcessFailureWake(pi, "process_disappeared", `No status update for ${Math.round(staleSeconds)}s and pid ${state.childPid} is gone.`);
  }
}

async function sendProcessFailureWake(pi: ExtensionAPI, reason: string, details: string): Promise<void> {
  if (!state || state.failureWakeSent) {
    return;
  }
  state.failureWakeSent = true;
  state.wakeInFlight = true;
  wakeToolTouched = false;
  const output = state.processOutput;
  const stdoutTail = output ? output.stdoutTail.join("") : "No captured stdout tail.";
  const stderrTail = output ? output.stderrTail.join("") : "No captured stderr tail.";
  const message = [
    `Triforce training process failure: ${reason}`,
    "",
    details,
    "",
    `Experiment: ${state.experimentDir}`,
    `Run: ${state.runDir ?? "unknown"}`,
    output ? `Stdout log: ${output.stdoutPath}` : "Stdout log: unavailable after reattach",
    output ? `Stderr log: ${output.stderrPath}` : "Stderr log: unavailable after reattach",
    "",
    "## stdout tail",
    "```text",
    stdoutTail.trimEnd(),
    "```",
    "",
    "## stderr tail",
    "```text",
    stderrTail.trimEnd(),
    "```",
    "",
    "Use the triforce-experiment skill decision loop. Inspect the logs, decide whether to edit/restart from checkpoint or finish, and update journal.md.",
  ].join("\n");
  await pi.sendMessage({
    customType: "triforce-training-process-failure",
    content: message,
    display: true,
    attribution: "user",
  }, { deliverAs: "nextTurn", triggerTurn: true });
}

async function stopChildForRestart(current: ExperimentState): Promise<void> {
  if (current.runDir) {
    writeControl(current, "stop", "restart requested by agent");
  }
  if (!current.child && current.childPid && isProcessAlive(current.childPid)) {
    await delay(120_000);
    if (isProcessAlive(current.childPid)) {
      try {
        process.kill(current.childPid, "SIGTERM");
      } catch {}
    }
    await delay(30_000);
    if (isProcessAlive(current.childPid)) {
      try {
        process.kill(current.childPid, "SIGKILL");
      } catch {}
    }
    return;
  }
  if (!current.child) {
    return;
  }
  const exitPromise = waitForExit(current.child);
  const timeoutPromise = delay(120_000).then(() => "timeout" as const);
  const result = await Promise.race([exitPromise, timeoutPromise]);
  if (result === "timeout" && current.child.pid) {
    current.child.kill("SIGTERM");
    const killTimeout = delay(30_000).then(() => "timeout" as const);
    const killResult = await Promise.race([exitPromise, killTimeout]);
    if (killResult === "timeout") {
      current.child.kill("SIGKILL");
    }
  }
}

function waitForExit(child: ChildProcess): Promise<"exit"> {
  const { promise, resolve } = Promise.withResolvers<"exit">();
  child.once("exit", () => resolve("exit"));
  return promise;
}

function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

function makeProcessOutput(experimentDir: string, pid: number): ProcessOutput {
  fs.mkdirSync(experimentDir, { recursive: true });
  const suffix = pid > 0 ? String(pid) : "unknown";
  return {
    stdoutPath: path.join(experimentDir, `train-${suffix}.stdout.log`),
    stderrPath: path.join(experimentDir, `train-${suffix}.stderr.log`),
    stdoutTail: [],
    stderrTail: [],
  };
}

function appendProcessOutput(kind: "stdout" | "stderr", text: string): void {
  if (!state?.processOutput) {
    return;
  }
  const output = state.processOutput;
  const targetPath = kind === "stdout" ? output.stdoutPath : output.stderrPath;
  fs.appendFileSync(targetPath, text, "utf8");
  const tail = kind === "stdout" ? output.stdoutTail : output.stderrTail;
  tail.push(text);
  while (tail.join("").length > 8_000 && tail.length > 1) {
    tail.shift();
  }
}

function readUpdatedAt(status: TriforceStatus | null): number | null {
  if (typeof status?.updated_at === "number") {
    return status.updated_at;
  }
  if (!state?.runDir) {
    return null;
  }
  const statusPath = path.join(state.runDir, "status.json");
  try {
    return fs.statSync(statusPath).mtimeMs / 1000;
  } catch {
    return null;
  }
}

function writeControl(current: ExperimentState, command: string, reason: string): void {
  if (!current.runDir) {
    return;
  }
  const payload = {
    schema_version: 1,
    command,
    reason,
    updated_at: Date.now() / 1000,
  } satisfies Record<string, unknown>;
  fs.writeFileSync(path.join(current.runDir, "control.json"), `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

function ensureJournal(experimentDir: string, params: StartParams): void {
  fs.mkdirSync(experimentDir, { recursive: true });
  const journalPath = path.join(experimentDir, "journal.md");
  if (fs.existsSync(journalPath)) {
    return;
  }
  const lines = [
    `# Triforce experiment: ${path.basename(experimentDir)}`,
    "",
    "## Goal",
    "",
    "To be filled by the agent before launch.",
    "",
    "## Initial scope",
    "",
    `- Started: ${new Date().toISOString()}`,
    `- Scenario: ${params.scenario}`,
    `- Action space: ${params.action_space}`,
    `- Model kind: ${params.model_kind}`,
    `- Baseline: ${params.baseline_eval_json ?? "none"}`,
    "- Wall-clock failsafe: 7 days",
    "",
  ];
  fs.writeFileSync(journalPath, lines.join("\n"), "utf8");
}

function toolResult(text: string, details: Record<string, unknown>, isError: boolean) {
  return {
    content: [{ type: "text", text }],
    details,
    isError,
  };
}

function isTriforceRoot(cwd: string): boolean {
  return fs.existsSync(path.join(cwd, "train.py")) && fs.existsSync(path.join(cwd, "triforce", "triforce.yaml"));
}

function sanitizeExperimentId(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, "-");
}

function timestampSlug(): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

function parseOutputRunDir(output: string): string | null {
  for (const line of output.split(/\r?\n/)) {
    if (line.startsWith("Output: ")) {
      return line.slice("Output: ".length).trim();
    }
  }
  return null;
}

function parseJsonObject(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

function readJsonFile(filePath: string): unknown {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as unknown;
  } catch {
    return null;
  }
}

function readTextIfExists(filePath: string): string | null {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isMilestoneEvent(value: unknown): value is MilestoneEvent {
  if (!isRecord(value)) {
    return false;
  }
  return value.type === "milestone"
    && typeof value.schema_version === "number"
    && typeof value.id === "number"
    && typeof value.timestamp === "number"
    && typeof value.reason === "string"
    && typeof value.run_dir === "string"
    && typeof value.experiment_dir === "string"
    && typeof value.step === "number"
    && typeof value.report_path === "string";
}

function isStatus(value: unknown): value is TriforceStatus {
  return isRecord(value) && typeof value.state === "string";
}

function readStringField(value: unknown, key: string): string | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const candidate = value[key];
  return typeof candidate === "string" ? candidate : undefined;
}

function readBooleanField(value: unknown, key: string): boolean | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const candidate = value[key];
  return typeof candidate === "boolean" ? candidate : undefined;
}

function safeReadDir(dirPath: string): string[] {
  try {
    return fs.readdirSync(dirPath);
  } catch {
    return [];
  }
}

function fileSize(filePath: string): number {
  try {
    return fs.statSync(filePath).size;
  } catch {
    return 0;
  }
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function wakeInProgress(): boolean {
  return state?.wakeInFlight ?? false;
}

function formatInt(value: unknown): string {
  return typeof value === "number" ? Math.round(value).toLocaleString() : "0";
}

function formatPct(value: unknown): string {
  return typeof value === "number" ? `${value.toFixed(1)}%` : "0.0%";
}

function formatNumber(value: unknown): string {
  return typeof value === "number" ? value.toFixed(1) : "n/a";
}

function formatUnknown(value: unknown): string {
  return typeof value === "number" ? value.toFixed(4) : "n/a";
}

function calculateLegEta(current: CurrentStatus, overall: OverallStatus): number | null {
  if (typeof current.steps !== "number" || typeof current.total_steps !== "number" || typeof overall.sps !== "number") {
    return null;
  }
  if (overall.sps <= 0 || current.steps >= current.total_steps) {
    return null;
  }
  return (current.total_steps - current.steps) / overall.sps;
}

function formatEta(value: unknown): string {
  if (typeof value !== "number") {
    return "n/a";
  }
  const minutes = Math.ceil(value / 60);
  if (minutes < 60) {
    return `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h${String(minutes % 60).padStart(2, "0")}m`;
}
