import os


for i in os.listdir(r'D:\Downloads\Video_Audio_ravdess'): #Folder where your MP4s are stored
    command=r"ffmpeg -i D:\Downloads\Video_Audio_ravdess\{} -ac 2 -f wav C:\Users\Vrittik\Desktop\Mini\Ravdess_123\{}.wav".format(i,i[:20])
    os.system(command)
    

    
    
    

            
            
