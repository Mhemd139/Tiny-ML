import numpy as np 
import sounddevice as sd
from scipy import signal
import tflite_runtime.interpreter as tflite

import RPi.GPIO as GPIO
from time import sleep




GPIO.setmode(GPIO.BOARD)
green_pin = 11  # pin number for green color
red_pin = 13  # pin number for red color

GPIO.setup(green_pin, GPIO.OUT)
GPIO.setup(red_pin, GPIO.OUT)

def logger (variable_name , variable_value):
    print(variable_name , ":" , variable_value)

def main():
    labels = ['green', 'off', 'on', 'red']
    duration = 1  # seconds
    print("Speak Now")
    fs = 44100  # samples per second
    audio_rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    int_audio = (np.clip(audio_rec , -32767 , 32767)) * 32767
    int_audio = int_audio.astype(np.int16)
    int_audio = np.squeeze(int_audio , axis = 1)


    ### Producing Spectrogram
    f, t, spec = signal.stft(int_audio, fs=44100, nperseg=255, noverlap = 124, nfft=256)
    spec = np.abs(spec)
    #input_data = np.reshape(spec , (1,1 , 129 , 338))

    input_data = np.reshape(spec , (1,1 , spec.shape[0] , spec.shape[1]))
    #logger("Input_Data Shape "  , input_data.shape)


    ### Model Loading 
    interpreter = tflite.Interpreter('Model/audio_led_model_3.tflite')

    input_details   = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    

    ## Model Predicting
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'] , input_data)
    interpreter.invoke()

    tflite_prediction_result = interpreter.get_tensor(output_details[0]['index'])
    label_index = np.argmax(tflite_prediction_result)
    logger("Lite Model Predictions ",labels[label_index]) #  labels[np.argmax(tflite_prediction_result)]

      
    if(label_index == 0 ):
        GPIO.output(green_pin, GPIO.LOW)  # turn off green color
        GPIO.output(red_pin, GPIO.LOW)   # turn off red color
    elif(label_index == 1):
        GPIO.output(green_pin, GPIO.LOW)   # turn off green color (Its supposed to be HIGH HIGH as 1 resembles on but I have only 1 bi color led)
        GPIO.output(red_pin, GPIO.HIGH)  # turn on red color         
    elif(label_index == 2):
        GPIO.output(green_pin, GPIO.HIGH)   # turn on green color
        GPIO.output(red_pin, GPIO.LOW)   # turn off red color
    elif(label_index == 3):
        GPIO.output(green_pin, GPIO.LOW)   # turn off green color
        GPIO.output(red_pin, GPIO.HIGH)   # turn on red color











if __name__  == '__main__':
    main()