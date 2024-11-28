#funcio que dibuixa les ones 
def plot_sound(path):
    plt.figure(figsize=(14, 5))
    x, sr = librosa.load(path)
    print("length {}, sample-rate {}".format(x.shape, sr))
    librosa.display.waveplot(x, sr=sr)
    
    return x

cami_exemple = "ACproject-14-grup\datasets\Data1\genres_original\blues\blues.00000.wav"
blues_audio = plot_sound(cami_exemple)
ipd.Audio(cami_exemple)