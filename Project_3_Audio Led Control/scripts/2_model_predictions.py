import numpy as np 
import sounddevice as sd
import numpy as np
import scipy
import tensorflow as tf
from scipy import signal




def logger (variable_name , variable_value):
    print(variable_name , ":" , variable_value)

def main():
    labels = ['green', 'red', 'on', 'off']
    ### Reading audio from mic
    duration = 1  # seconds
    print("Speak Now")
    fs = 22050  # samples per second
    audio_rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    int_audio = (np.clip(audio_rec , -32767 , 32767)) * 32767
    int_audio = int_audio.astype(np.int16)
    int_audio = np.squeeze(int_audio , axis = 1)


    ### Producing Spectrogram
    f, t, spec = signal.stft(int_audio, fs=22050, nperseg=255, noverlap = 124, nfft=256)
    spec = np.abs(spec)
    input_data = np.reshape(spec , (1 ,1 , spec.shape[0] , spec.shape[1]))
    #logger("Input_Data Shape "  , input_data.shape)


    ### Model Loading 
    interpreter = tf.lite.Interpreter('data/Model/audio_led_model_3.tflite')

    input_details   = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    

    ## Model Predicting
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'] , input_data)
    interpreter.invoke()

    tflite_prediction_result = interpreter.get_tensor(output_details[0]['index'])
    logger("Lite Model Predictions ",labels[np.argmax(tflite_prediction_result)]) #  labels[np.argmax(tflite_prediction_result)]

      

if __name__  == '__main__':
    main()