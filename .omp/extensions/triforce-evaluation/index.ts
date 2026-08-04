import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

interface EvaluationState {
  evalId: string;
  evalDir: string;
  modelPath: string;
  scenario: string;
  episodes: number;
  reprocess: boolean;
  compare: [string, string] | null;
  childPid: number | null;
  child: ChildProcess | null;
  processOutput: ProcessOutput;
  startedAt: number;
  completedAt: number | null;
  resultJsonPath: string | null;
  resultMdPath: string | null;
  compareOutputPath: string | null;
  statusPath: string;
  wakeSent: boolean;
  widgetTimer: NodeJS.Timeout | null;
}

interface ProcessOutput {
  stdoutPath: string;
  stderrPath: string;
  stdoutTail: string[];
  stderrTail: string[];
}

interface EvaluationStartParams {
  eval_id?: string;
  model_path: string;
  scenario: string;
  episodes: number;
  reprocess: boolean;
  render: boolean;
  limit: number;
  frame_stack: number;
  steps?: number[];
  extra_args: string[];
}

interface EvaluationCompareParams {
  eval_id?: string;
  baseline_eval_json: string;
  candidate_eval_json: string;
}

interface EvalSummary {
  episodes?: number;
  scenario?: string;
  successRate?: number | null;
  maxProgressReached?: number;
  maxProgressPercent?: number;
}

interface UiHooks {
  hasUI: () => boolean;
  setWidget: (lines: string[]) => void | Promise<void>;
}

let state: EvaluationState | null = null;
let uiHooks: UiHooks | null = null;

export default function triforceEvaluation(pi: ExtensionAPI): void {
  const z = pi.zod;
  pi.setLabel("Triforce Evaluations");

  pi.on("session_start", async (_event, ctx) => {
    uiHooks = {
      hasUI: () => ctx.hasUI,
      setWidget: (lines: string[]) => ctx.ui.setWidget("aboveEditor", lines),
    };
  });

  pi.on("session_shutdown", async () => {
    if (state?.child && state.childPid && isProcessAlive(state.childPid)) {
      state.child.kill("SIGTERM");
    }
    stopWidgetTimer();
  });

  pi.registerTool({
    name: "triforce_evaluation_start",
    label: "Start Triforce Evaluation",
    description: "Run a long Triforce evaluate.py job through OMP and wake only when it completes or fails.",
    parameters: z.object({
      eval_id: z.string().optional(),
      model_path: z.string(),
      scenario: z.string(),
      episodes: z.number().int().positive().default(50),
      reprocess: z.boolean().default(false),
      render: z.boolean().default(false),
      limit: z.number().int().default(-1),
      frame_stack: z.number().int().positive().default(3),
      steps: z.array(z.number().int()).optional(),
      extra_args: z.array(z.string()).default([]),
    }),
    async execute(_toolCallId, params) {
      const parsed = normalizeStartParams(params);
      const validation = validateLaunch(parsed.model_path);
      if (validation) {
        return toolResult(validation, {}, true);
      }
      const result = startEvaluation(pi, parsed);
      return toolResult(result.message, result.details, !result.ok);
    },
  });

  pi.registerTool({
    name: "triforce_evaluation_compare",
    label: "Compare Triforce Evaluations",
    description: "Run evaluate.py --compare through OMP and wake only when comparison completes or fails.",
    parameters: z.object({
      eval_id: z.string().optional(),
      baseline_eval_json: z.string(),
      candidate_eval_json: z.string(),
    }),
    async execute(_toolCallId, params) {
      const parsed = normalizeCompareParams(params);
      const validation = validateCompareLaunch(parsed.baseline_eval_json, parsed.candidate_eval_json);
      if (validation) {
        return toolResult(validation, {}, true);
      }
      const result = startComparison(pi, parsed);
      return toolResult(result.message, result.details, !result.ok);
    },
  });

  pi.registerTool({
    name: "triforce_evaluation_status",
    label: "Triforce Evaluation Status",
    description: "Return the active or latest Triforce evaluation plugin status.",
    parameters: z.object({}),
    async execute() {
      return toolResult("Triforce evaluation status loaded.", {
        active: state !== null,
        status: readStatusForTool(),
      }, false);
    },
  });

  pi.registerTool({
    name: "triforce_evaluation_cancel",
    label: "Cancel Triforce Evaluation",
    description: "Cancel the active Triforce evaluation without waking the LLM.",
    parameters: z.object({ reason: z.string().default("") }),
    async execute(_toolCallId, params) {
      const reason = readStringField(params, "reason") ?? "";
      if (!state) {
        return toolResult("No active Triforce evaluation to cancel.", {}, true);
      }
      await cancelEvaluation(reason);
      return toolResult("Triforce evaluation cancelled.", { eval_id: state?.evalId ?? null }, false);
    },
  });
}

function startEvaluation(pi: ExtensionAPI, params: EvaluationStartParams) {
  const pythonPath = path.join(process.cwd(), ".venv", "bin", "python");
  const evalId = sanitizeExperimentId(params.eval_id ?? `eval-${params.scenario}-${timestampSlug()}`);
  const evalDir = path.join("training", "evaluations", evalId);
  fs.mkdirSync(evalDir, { recursive: true });
  const args = ["evaluate.py", params.model_path, params.scenario, "--episodes", String(params.episodes), "--frame-stack", String(params.frame_stack)];
  if (params.reprocess) {
    args.push("--reprocess");
  }
  if (params.render) {
    args.push("--render");
  }
  if (params.limit !== -1) {
    args.push("--limit", String(params.limit));
  }
  if (params.steps && params.steps.length > 0) {
    args.push("--steps", ...params.steps.map(String));
  }
  args.push(...params.extra_args);

  const child = spawn(pythonPath, args, {
    cwd: process.cwd(),
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const processOutput = makeProcessOutput(evalDir, child.pid ?? 0);
  const expected = expectedResultPaths(params.model_path);
  state = {
    evalId,
    evalDir,
    modelPath: params.model_path,
    scenario: params.scenario,
    episodes: params.episodes,
    reprocess: params.reprocess,
    compare: null,
    childPid: child.pid ?? null,
    child,
    processOutput,
    startedAt: Date.now() / 1000,
    completedAt: null,
    resultJsonPath: expected.primaryJson,
    resultMdPath: expected.primaryMd,
    compareOutputPath: null,
    statusPath: path.join(evalDir, "status.json"),
    wakeSent: false,
    widgetTimer: null,
  };
  writeStatus("running", { command: [pythonPath, ...args], expected_eval_json_path: expected.primaryJson, expected_eval_md_path: expected.primaryMd });
  wireChild(pi, child, false);
  child.unref();
  startWidgetTimer();
  return {
    ok: true,
    message: "Triforce evaluation started.",
    details: { eval_id: evalId, eval_dir: evalDir, child_pid: child.pid ?? null, status_path: state.statusPath },
  };
}

function startComparison(pi: ExtensionAPI, params: EvaluationCompareParams) {
  const pythonPath = path.join(process.cwd(), ".venv", "bin", "python");
  const evalId = sanitizeExperimentId(params.eval_id ?? `compare-${timestampSlug()}`);
  const evalDir = path.join("training", "evaluations", evalId);
  fs.mkdirSync(evalDir, { recursive: true });
  const args = ["evaluate.py", "--compare", params.baseline_eval_json, params.candidate_eval_json];
  const child = spawn(pythonPath, args, {
    cwd: process.cwd(),
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  state = {
    evalId,
    evalDir,
    modelPath: params.candidate_eval_json,
    scenario: "compare",
    episodes: 0,
    reprocess: false,
    compare: [params.baseline_eval_json, params.candidate_eval_json],
    childPid: child.pid ?? null,
    child,
    processOutput: makeProcessOutput(evalDir, child.pid ?? 0),
    startedAt: Date.now() / 1000,
    completedAt: null,
    resultJsonPath: null,
    resultMdPath: null,
    compareOutputPath: path.join(evalDir, "compare.md"),
    statusPath: path.join(evalDir, "status.json"),
    wakeSent: false,
    widgetTimer: null,
  };
  writeStatus("running", { command: [pythonPath, ...args], compare_output_path: state.compareOutputPath });
  wireChild(pi, child, true);
  child.unref();
  startWidgetTimer();
  return {
    ok: true,
    message: "Triforce evaluation comparison started.",
    details: { eval_id: evalId, eval_dir: evalDir, child_pid: child.pid ?? null, status_path: state.statusPath },
  };
}

function wireChild(pi: ExtensionAPI, child: ChildProcess, compare: boolean): void {
  child.stdout?.on("data", (chunk: Buffer) => appendProcessOutput("stdout", chunk.toString("utf8")));
  child.stderr?.on("data", (chunk: Buffer) => appendProcessOutput("stderr", chunk.toString("utf8")));
  child.on("exit", (code, signal) => {
    if (!state) {
      return;
    }
    state.child = null;
    state.completedAt = Date.now() / 1000;
    if (code === 0 && !signal) {
      completeEvaluation(pi, compare, code, signal);
    } else {
      failEvaluation(pi, code, signal, "Evaluation process failed.");
    }
  });
}

function completeEvaluation(pi: ExtensionAPI, compare: boolean, code: number | null, signal: NodeJS.Signals | null): void {
  if (!state || state.wakeSent) {
    return;
  }
  if (compare && state.compareOutputPath) {
    const stdout = readTextIfExists(state.processOutput.stdoutPath) ?? "";
    fs.writeFileSync(state.compareOutputPath, stdout, "utf8");
    writeStatus("complete", { exit_code: code, signal, compare_output_path: state.compareOutputPath });
    void sendCompletionWake(pi, "Comparison complete.");
    return;
  }
  const resultPaths = discoverResultPaths(state.modelPath);
  const primary = resultPaths[0] ?? null;
  state.resultJsonPath = primary?.jsonPath ?? state.resultJsonPath;
  state.resultMdPath = primary?.mdPath ?? state.resultMdPath;
  const summary = primary ? summarizeEvalJson(primary.jsonPath) : null;
  const finalState = primary ? "complete" : "complete_no_results";
  writeStatus(finalState, {
    exit_code: code,
    signal,
    result_json_paths: resultPaths.map(result => result.jsonPath),
    result_md_paths: resultPaths.map(result => result.mdPath),
    summary,
  });
  const message = primary
    ? "Evaluation complete."
    : "Evaluation exited successfully but no .eval.json result was found. Ask the user whether to re-run through the plugin. Do not run evaluate.py directly.";
  void sendCompletionWake(pi, message);
}

function failEvaluation(pi: ExtensionAPI, code: number | null, signal: NodeJS.Signals | null, details: string): void {
  if (!state || state.wakeSent) {
    return;
  }
  writeStatus("failed", {
    exit_code: code,
    signal,
    stdout_tail: state.processOutput.stdoutTail.join(""),
    stderr_tail: state.processOutput.stderrTail.join(""),
  });
  void sendFailureWake(pi, details);
}

async function sendCompletionWake(pi: ExtensionAPI, headline: string): Promise<void> {
  if (!state || state.wakeSent) {
    return;
  }
  state.wakeSent = true;
  clearWidget();
  const status = readJsonFile(state.statusPath);
  const summary = isRecord(status) && isRecord(status.summary) ? renderSummary(status.summary) : "No parsed summary.";
  const message = [
    `Triforce evaluation complete: ${state.evalId}`,
    "",
    headline,
    "",
    `Model: ${state.modelPath}`,
    `Scenario: ${state.scenario}`,
    `Episodes: ${state.episodes}`,
    `Status: ${state.statusPath}`,
    `Stdout log: ${state.processOutput.stdoutPath}`,
    `Stderr log: ${state.processOutput.stderrPath}`,
    `Result JSON: ${state.resultJsonPath ?? "none"}`,
    `Result Markdown: ${state.resultMdPath ?? "none"}`,
    `Compare output: ${state.compareOutputPath ?? "none"}`,
    "",
    "## Summary",
    summary,
    "",
    "Use the triforce-evaluation skill. Summarize the result or append it to summary.md if this was part of a training experiment.",
  ].join("\n");
  await pi.sendMessage({
    customType: "triforce-evaluation-complete",
    content: message,
    display: true,
    attribution: "user",
  }, { deliverAs: "nextTurn", triggerTurn: true });
}

async function sendFailureWake(pi: ExtensionAPI, details: string): Promise<void> {
  if (!state || state.wakeSent) {
    return;
  }
  state.wakeSent = true;
  clearWidget();
  const message = [
    `Triforce evaluation failed: ${state.evalId}`,
    "",
    details,
    "",
    `Model: ${state.modelPath}`,
    `Scenario: ${state.scenario}`,
    `Status: ${state.statusPath}`,
    `Stdout log: ${state.processOutput.stdoutPath}`,
    `Stderr log: ${state.processOutput.stderrPath}`,
    "",
    "## stdout tail",
    "```text",
    state.processOutput.stdoutTail.join("").trimEnd(),
    "```",
    "",
    "## stderr tail",
    "```text",
    state.processOutput.stderrTail.join("").trimEnd(),
    "```",
    "",
    "Ask the user to fix the evaluation plugin/environment problem. Do not run evaluate.py directly.",
  ].join("\n");
  await pi.sendMessage({
    customType: "triforce-evaluation-failure",
    content: message,
    display: true,
    attribution: "user",
  }, { deliverAs: "nextTurn", triggerTurn: true });
}

async function cancelEvaluation(reason: string): Promise<void> {
  if (!state) {
    return;
  }
  const child = state.child;
  if (child && state.childPid && isProcessAlive(state.childPid)) {
    child.kill("SIGTERM");
    const exitPromise = waitForExit(child);
    const timeoutPromise = delay(30_000).then(() => "timeout" as const);
    const result = await Promise.race([exitPromise, timeoutPromise]);
    if (result === "timeout" && state.childPid && isProcessAlive(state.childPid)) {
      child.kill("SIGKILL");
    }
  }
  writeStatus("cancelled", { reason });
  clearWidget();
}

function startWidgetTimer(): void {
  if (!state || state.widgetTimer) {
    return;
  }
  state.widgetTimer = setInterval(updateWidget, 30_000);
  updateWidget();
}

function stopWidgetTimer(): void {
  if (!state?.widgetTimer) {
    return;
  }
  clearInterval(state.widgetTimer);
  state.widgetTimer = null;
}

function updateWidget(): void {
  if (!state || !(uiHooks?.hasUI() ?? false)) {
    return;
  }
  const lastStdout = lastNonEmptyLine(state.processOutput.stdoutTail.join(""));
  const lines = [
    `Triforce eval: ${statusState()} pid=${state.childPid ?? "n/a"}`,
    `Model: ${path.basename(state.modelPath)}`,
    `Scenario: ${state.scenario}`,
    `Episodes: ${state.episodes}`,
    `Elapsed: ${formatDuration(Date.now() / 1000 - state.startedAt)}`,
    `Output: ${state.evalDir}`,
    `Stdout: ${lastStdout ?? "n/a"}`,
    `Result: ${state.resultJsonPath ? path.basename(state.resultJsonPath) : "pending"}`,
  ];
  void uiHooks.setWidget(lines);
}

function clearWidget(): void {
  stopWidgetTimer();
  if (uiHooks?.hasUI()) {
    void uiHooks.setWidget([]);
  }
}

function writeStatus(status: string, extra: Record<string, unknown>): void {
  if (!state) {
    return;
  }
  const payload = {
    state: status,
    eval_id: state.evalId,
    eval_dir: state.evalDir,
    model_path: state.modelPath,
    scenario: state.scenario,
    episodes: state.episodes,
    reprocess: state.reprocess,
    compare: state.compare,
    pid: state.childPid,
    started_at: state.startedAt,
    completed_at: state.completedAt,
    stdout_path: state.processOutput.stdoutPath,
    stderr_path: state.processOutput.stderrPath,
    result_json_path: state.resultJsonPath,
    result_md_path: state.resultMdPath,
    compare_output_path: state.compareOutputPath,
    updated_at: Date.now() / 1000,
    ...extra,
  } satisfies Record<string, unknown>;
  fs.writeFileSync(state.statusPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

function readStatusForTool(): unknown {
  if (state) {
    return readJsonFile(state.statusPath) ?? { state: statusState(), eval_id: state.evalId };
  }
  const latest = latestStatusPath();
  return latest ? readJsonFile(latest) : { state: "inactive" };
}

function statusState(): string {
  if (!state) {
    return "inactive";
  }
  if (state.completedAt !== null) {
    const parsed = readJsonFile(state.statusPath);
    return isRecord(parsed) && typeof parsed.state === "string" ? parsed.state : "complete";
  }
  return "running";
}

function validateLaunch(modelPath: string): string | null {
  const common = validateRootAndPython();
  if (common) {
    return common;
  }
  if (!fs.existsSync(modelPath)) {
    return `Model path does not exist: ${modelPath}`;
  }
  return null;
}

function validateCompareLaunch(baselinePath: string, candidatePath: string): string | null {
  const common = validateRootAndPython();
  if (common) {
    return common;
  }
  if (!fs.existsSync(baselinePath)) {
    return `Evaluation JSON path does not exist: ${baselinePath}`;
  }
  if (!fs.existsSync(candidatePath)) {
    return `Evaluation JSON path does not exist: ${candidatePath}`;
  }
  return null;
}

function validateRootAndPython(): string | null {
  if (!isTriforceRoot(process.cwd())) {
    return "Run OMP from the Triforce repository root.";
  }
  const pythonPath = path.join(process.cwd(), ".venv", "bin", "python");
  if (!fs.existsSync(pythonPath)) {
    return "Missing .venv/bin/python. Ask the user to fix the Triforce Python environment; do not run evaluate.py directly.";
  }
  return null;
}

function expectedResultPaths(modelPath: string): { primaryJson: string | null; primaryMd: string | null } {
  if (fs.existsSync(modelPath) && fs.statSync(modelPath).isFile() && modelPath.endsWith(".pt")) {
    return { primaryJson: modelPath.replace(/\.pt$/, ".eval.json"), primaryMd: modelPath.replace(/\.pt$/, ".eval.md") };
  }
  return { primaryJson: null, primaryMd: null };
}

function discoverResultPaths(modelPath: string): Array<{ jsonPath: string; mdPath: string }> {
  if (fs.existsSync(modelPath) && fs.statSync(modelPath).isFile() && modelPath.endsWith(".pt")) {
    const paths = expectedResultPaths(modelPath);
    if (paths.primaryJson && fs.existsSync(paths.primaryJson)) {
      return [{ jsonPath: paths.primaryJson, mdPath: paths.primaryMd ?? paths.primaryJson.replace(/\.json$/, ".md") }];
    }
    return [];
  }
  const results: Array<{ jsonPath: string; mdPath: string }> = [];
  for (const filename of safeReadDir(modelPath)) {
    if (!filename.endsWith(".pt")) {
      continue;
    }
    const modelFile = path.join(modelPath, filename);
    const jsonPath = modelFile.replace(/\.pt$/, ".eval.json");
    if (fs.existsSync(jsonPath)) {
      results.push({ jsonPath, mdPath: modelFile.replace(/\.pt$/, ".eval.md") });
    }
  }
  results.sort((left, right) => left.jsonPath.localeCompare(right.jsonPath));
  return results;
}

function summarizeEvalJson(jsonPath: string): EvalSummary | null {
  const parsed = readJsonFile(jsonPath);
  if (!isRecord(parsed)) {
    return null;
  }
  const progressValues = Array.isArray(parsed.progress_values) ? parsed.progress_values : [];
  const maxProgress = typeof parsed.max_progress === "number" ? parsed.max_progress : null;
  const successCount = maxProgress === null ? 0 : progressValues.filter(value => typeof value === "number" && value >= maxProgress).length;
  const metrics = isRecord(parsed.metrics) ? parsed.metrics : {};
  const episodes = typeof parsed.episodes === "number" ? parsed.episodes : undefined;
  return {
    episodes,
    scenario: typeof parsed.scenario === "string" ? parsed.scenario : undefined,
    successRate: typeof metrics["success-rate"] === "number" ? metrics["success-rate"] : null,
    maxProgressReached: successCount,
    maxProgressPercent: episodes && episodes > 0 ? successCount / episodes : undefined,
  };
}

function renderSummary(summary: Record<string, unknown>): string {
  return [
    `- Episodes: ${summary.episodes ?? "unknown"}`,
    `- Scenario: ${summary.scenario ?? "unknown"}`,
    `- Success rate: ${typeof summary.successRate === "number" ? summary.successRate : "n/a"}`,
    `- Max-progress successes: ${summary.maxProgressReached ?? "unknown"}`,
    `- Max-progress percent: ${typeof summary.maxProgressPercent === "number" ? (summary.maxProgressPercent * 100).toFixed(1) : "n/a"}%`,
  ].join("\n");
}

function normalizeStartParams(value: unknown): EvaluationStartParams {
  return {
    eval_id: readStringField(value, "eval_id"),
    model_path: requireStringField(value, "model_path"),
    scenario: requireStringField(value, "scenario"),
    episodes: readNumberField(value, "episodes") ?? 50,
    reprocess: readBooleanField(value, "reprocess") ?? false,
    render: readBooleanField(value, "render") ?? false,
    limit: readNumberField(value, "limit") ?? -1,
    frame_stack: readNumberField(value, "frame_stack") ?? 3,
    steps: readNumberArrayField(value, "steps"),
    extra_args: readStringArrayField(value, "extra_args") ?? [],
  };
}

function normalizeCompareParams(value: unknown): EvaluationCompareParams {
  return {
    eval_id: readStringField(value, "eval_id"),
    baseline_eval_json: requireStringField(value, "baseline_eval_json"),
    candidate_eval_json: requireStringField(value, "candidate_eval_json"),
  };
}

function makeProcessOutput(evalDir: string, pid: number): ProcessOutput {
  fs.mkdirSync(evalDir, { recursive: true });
  const suffix = pid > 0 ? String(pid) : "unknown";
  return {
    stdoutPath: path.join(evalDir, `evaluate-${suffix}.stdout.log`),
    stderrPath: path.join(evalDir, `evaluate-${suffix}.stderr.log`),
    stdoutTail: [],
    stderrTail: [],
  };
}

function appendProcessOutput(kind: "stdout" | "stderr", text: string): void {
  if (!state) {
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

function latestStatusPath(): string | null {
  const root = path.join("training", "evaluations");
  let latest: { mtime: number; statusPath: string } | null = null;
  for (const evalName of safeReadDir(root)) {
    const statusPath = path.join(root, evalName, "status.json");
    if (!fs.existsSync(statusPath)) {
      continue;
    }
    const mtime = fs.statSync(statusPath).mtimeMs;
    if (!latest || mtime > latest.mtime) {
      latest = { mtime, statusPath };
    }
  }
  return latest?.statusPath ?? null;
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
  const pad = (part: number) => String(part).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
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

function readStringField(value: unknown, key: string): string | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const candidate = value[key];
  return typeof candidate === "string" ? candidate : undefined;
}

function requireStringField(value: unknown, key: string): string {
  const candidate = readStringField(value, key);
  if (!candidate) {
    throw new Error(`${key} is required.`);
  }
  return candidate;
}

function readBooleanField(value: unknown, key: string): boolean | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const candidate = value[key];
  return typeof candidate === "boolean" ? candidate : undefined;
}

function readNumberField(value: unknown, key: string): number | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const candidate = value[key];
  return typeof candidate === "number" && Number.isFinite(candidate) ? candidate : undefined;
}

function readStringArrayField(value: unknown, key: string): string[] | undefined {
  if (!isRecord(value) || !Array.isArray(value[key])) {
    return undefined;
  }
  const values = value[key];
  return values.every(item => typeof item === "string") ? values : undefined;
}

function readNumberArrayField(value: unknown, key: string): number[] | undefined {
  if (!isRecord(value) || !Array.isArray(value[key])) {
    return undefined;
  }
  const values = value[key];
  return values.every(item => typeof item === "number" && Number.isInteger(item)) ? values : undefined;
}

function safeReadDir(dirPath: string): string[] {
  try {
    return fs.readdirSync(dirPath);
  } catch {
    return [];
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

function lastNonEmptyLine(text: string): string | null {
  const lines = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  return lines.length > 0 ? lines[lines.length - 1] : null;
}

function formatDuration(seconds: number): string {
  const rounded = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(rounded / 60);
  const secs = rounded % 60;
  if (minutes < 60) {
    return `${minutes}m${String(secs).padStart(2, "0")}s`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h${String(minutes % 60).padStart(2, "0")}m`;
}
