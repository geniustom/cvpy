# -*- coding: utf-8 -*-

import speech_recognition
from pydub import AudioSegment
 
 
def trans_mp3_to_wav(filepath):
    song = AudioSegment.from_mp3(filepath)
    song.export("now.wav", format="wav")


trans_mp3_to_wav("voice_test.mp3")
r = speech_recognition.Recognizer()
with speech_recognition.AudioFile("now.wav") as source:
     r.adjust_for_ambient_noise(source, duration=0.5)
     audio = r.record(source)

trans=r.recognize_google(audio,language='zh-tw',show_all=True)
print(trans)