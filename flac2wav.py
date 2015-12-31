import os
import audiotools

root = '/home/benk/uni/shai/neuraltts/data/1919/142785'

for f in os.listdir(root):
    fullf = os.path.join(root, f)
    fullnewf = os.path.join(root, os.path.splitext(f)[0] + '.wav')
    audiotools.open(fullf).convert(fullnewf, audiotools.WaveAudio)
