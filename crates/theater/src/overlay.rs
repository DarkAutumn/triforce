//! The controller overlay: a small picture of the pad in the top right of a frame, over the HUD's `-LIFE-`,
//! whose d-pad arms and buttons light up while they are held.
//!
//! It is drawn into the RGB frame at the console's resolution, so it scales with the picture and a frame and
//! its buttons can never be out of step.  The pad's plate is a translucent grey, an idle button a darker one,
//! a held button opaque white.

use zcore::machine::{HEIGHT, WIDTH};

pub const A: u8 = 0x01;
pub const B: u8 = 0x02;
pub const SELECT: u8 = 0x04;
pub const START: u8 = 0x08;
pub const UP: u8 = 0x10;
pub const DOWN: u8 = 0x20;
pub const LEFT: u8 = 0x40;
pub const RIGHT: u8 = 0x80;

/// The plate: `-LIFE-` spans x 193..=229, y 24..=31; the hearts start at y 40.
pub const X: usize = 182;
pub const Y: usize = 14;
pub const PLATE_WIDTH: usize = 62;
pub const PLATE_HEIGHT: usize = 24;

const PLATE: ([u8; 3], u32) = ([118, 118, 126], 185);
const IDLE: ([u8; 3], u32) = ([34, 34, 40], 200);
pub const HELD: [u8; 3] = [255, 255, 255];

/// A part of the pad, in plate coordinates: a rectangle (`x`, `y`, `width`, `height`), a disc when `round`.
struct Part {
    button: u8,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    round: bool,
}

const fn bar(button: u8, x: usize, y: usize, width: usize, height: usize) -> Part {
    Part { button, x, y, width, height, round: false }
}

/// The d-pad is a cross around (12, 12); its hub belongs to no button.
const PARTS: [Part; 9] = [
    bar(UP, 10, 5, 5, 5),
    bar(DOWN, 10, 15, 5, 5),
    bar(LEFT, 5, 10, 5, 5),
    bar(RIGHT, 15, 10, 5, 5),
    bar(0, 10, 10, 5, 5),
    bar(SELECT, 24, 13, 7, 3),
    bar(START, 33, 13, 7, 3),
    Part { button: B, x: 43, y: 11, width: 7, height: 7, round: true },
    Part { button: A, x: 52, y: 11, width: 7, height: 7, round: true },
];

fn blend(pixel: &mut [u8], (color, alpha): ([u8; 3], u32)) {
    for (channel, tint) in pixel.iter_mut().zip(color) {
        *channel = ((u32::from(*channel) * (255 - alpha) + u32::from(tint) * alpha + 127) / 255) as u8;
    }
}

/// Draws the pad with `buttons` (the controller byte) held into `frame` (RGB, 240 rows of 256).
pub fn draw_controller(frame: &mut [u8], buttons: u8) {
    assert_eq!(frame.len(), WIDTH * HEIGHT * 3, "an RGB frame");
    let pixel = |x: usize, y: usize| 3 * ((Y + y) * WIDTH + X + x);
    for y in 0..PLATE_HEIGHT {
        for x in 0..PLATE_WIDTH {
            // The corners are cut.
            if (x == 0 || x == PLATE_WIDTH - 1) && (y == 0 || y == PLATE_HEIGHT - 1) {
                continue;
            }
            let at = pixel(x, y);
            blend(&mut frame[at..at + 3], PLATE);
        }
    }
    for part in &PARTS {
        let held = part.button & buttons != 0;
        for y in 0..part.height {
            for x in 0..part.width {
                // A disc of diameter 7: the square less its corner pixels.
                let (dx, dy) = (x.abs_diff(part.width / 2), y.abs_diff(part.height / 2));
                if part.round && dx + dy > 4 {
                    continue;
                }
                let at = pixel(part.x + x, part.y + y);
                if held {
                    frame[at..at + 3].copy_from_slice(&HELD);
                } else {
                    blend(&mut frame[at..at + 3], IDLE);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(fill: u8) -> Vec<u8> {
        vec![fill; WIDTH * HEIGHT * 3]
    }

    fn at(frame: &[u8], x: usize, y: usize) -> [u8; 3] {
        let i = 3 * (y * WIDTH + x);
        [frame[i], frame[i + 1], frame[i + 2]]
    }

    /// The middle pixel of each button's part, in frame coordinates.
    fn middle(button: u8) -> (usize, usize) {
        let part = PARTS.iter().find(|part| part.button == button).unwrap();
        (X + part.x + part.width / 2, Y + part.y + part.height / 2)
    }

    #[test]
    fn only_held_buttons_are_white_and_nothing_outside_the_plate_changes() {
        let buttons = [A, B, SELECT, START, UP, DOWN, LEFT, RIGHT];
        for held in [0u8, A, UP | B, LEFT | START, 0xFF] {
            let mut drawn = frame(0);
            draw_controller(&mut drawn, held);
            for button in buttons {
                let (x, y) = middle(button);
                assert_eq!(at(&drawn, x, y) == HELD, held & button != 0, "button {button:#04x} of {held:#04x}");
            }
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let inside = (X..X + PLATE_WIDTH).contains(&x) && (Y..Y + PLATE_HEIGHT).contains(&y);
                    assert!(inside || at(&drawn, x, y) == [0, 0, 0], "({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn the_plate_is_translucent_and_stays_clear_of_the_hearts() {
        // Over white the plate is a light grey, over black a dark one: the picture shows through.
        let (mut white, mut black) = (frame(255), frame(0));
        draw_controller(&mut white, 0);
        draw_controller(&mut black, 0);
        let (over_white, over_black) = (at(&white, X + 2, Y + 2), at(&black, X + 2, Y + 2));
        assert!(over_white[0] > over_black[0] + 60 && over_white[0] < 255 && over_black[0] > 0);
        // An idle button is darker than the plate around it, and never white.
        let (x, y) = middle(A);
        assert!(at(&white, x, y)[0] < over_white[0] && at(&white, x, y) != HELD);
        // It covers -LIFE- (x 193..=229, y 24..=31) and ends above the hearts (y 40).
        assert!(X <= 193 && X + PLATE_WIDTH > 229 && Y <= 24 && Y + PLATE_HEIGHT > 31 && Y + PLATE_HEIGHT <= 40);
    }
}
