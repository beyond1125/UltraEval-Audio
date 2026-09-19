# ADDED BY US. Upstream `Speech2text` asserts its input is an existing audio file. A full-duplex
# model can legitimately choose never to speak; its S2S adapter then returns `audio: ""`. This
# subclass maps exactly that empty string to an empty transcript WITHOUT calling the ASR model, so
# the item is scored (as wrong) inside the full denominator -- the same way the S2T arm scores an
# empty text answer -- instead of being counted as a failure. Any other input takes upstream's path
# unchanged, including its assertion for a non-empty path that does not exist.
from audio_evals.process.speech import Speech2text


class Speech2textAllowEmpty(Speech2text):
    def __call__(self, answer: str) -> str:
        if answer == "":
            return ""
        return super().__call__(answer)
