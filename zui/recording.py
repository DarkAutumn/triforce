import os
import cv2
import tqdm
import pygame

class Recording:
    """Used to track and save a recording of the game."""
    # pylint: disable=no-member
    def __init__(self, dimensions, buffer_size):
        self.dimensions = dimensions
        self.recording = None
        self.buffer_size = buffer_size
        self.buffer = []

    def _get_recording(self):
        if self.recording is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.recording = cv2.VideoWriter(self.__get_filename(), fourcc, 60.1, self.dimensions)

        return self.recording

    def write(self, surface):
        """Adds a frame to the recording."""
        surface = surface.copy()

        if self.buffer_size <= 1:
            self._write_surface(surface.copy())

        else:
            self.buffer.append(surface)
            if len(self.buffer) >= self.buffer_size:
                self.flush()

    def _write_surface(self, surface):
        result_frame = pygame.surfarray.array3d(surface)
        result_frame = result_frame.transpose([1, 0, 2])
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)

        recording = self._get_recording()
        recording.write(result_frame)

    def clear(self):
        """Clears the buffer without writing it out."""
        self.buffer.clear()

    def flush(self):
        """Writes the buffer to the recording."""
        if len(self.buffer) < 1000:
            for frame in self.buffer:
                self._write_surface(frame)
        else:
            for frame in tqdm.tqdm(self.buffer):
                self._write_surface(frame)

        self.buffer.clear()

    def close(self, write):
        """Stops the recording."""
        if write:
            self.flush()
        if self.recording:
            self.recording.release()
            self.recording = None

    def __get_filename(self):
        directory = os.path.join(os.getcwd(), "recording")
        if not os.path.exists(directory):
            os.makedirs(directory)

        i = 0
        while True:
            filename = os.path.join(directory, f"gameplay_{i:03d}.avi")
            if not os.path.exists(filename):
                return filename
            i += 1
