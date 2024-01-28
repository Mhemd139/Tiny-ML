#This file was created as a result of training my model on a different microphone than the one connected to my pi which resulted in a confusion thats why im using the files ive trained my model on to represent the model working correctly  
import numpy as np
import sounddevice as sd
from scipy import signal
import tflite_runtime.interpreter as tflite
import scipy.io.wavfile as wavfile
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)
green_pin = 11  # pin number for green color
red_pin = 13  # pin number for red color

GPIO.setup(green_pin, GPIO.OUT)
GPIO.setup(red_pin, GPIO.OUT)

def logger(variable_name, variable_value):
    print(variable_name, ":", variable_value)

def main():
    labels = ['green', 'off', 'on', 'red']
    wav_file_path = 'data/green/green_0.wav'

    fs, audio_rec = wavfile.read(wav_file_path)
    audio_rec = audio_rec.astype(np.int16)

    ### Producing Spectrogram
    f, t, spec = signal.stft(audio_rec, fs=fs, nperseg=255, noverlap=124, nfft=256)
    spec = np.abs(spec)
    input_data = np.reshape(spec, (1, 1, spec.shape[0], spec.shape[1]))

    ### Model Loading
    interpreter = tflite.Interpreter('Model/audio_led_model_3.tflite')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ## Model Predicting
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    tflite_prediction_result = interpreter.get_tensor(output_details[0]['index'])
    label_index = np.argmax(tflite_prediction_result)
    logger("Lite Model Predictions ", labels[label_index])

    if label_index == 0:
        GPIO.output(green_pin, GPIO.HIGH)  # turn on green color
        GPIO.output(red_pin, GPIO.LOW)   # turn off red color
    elif label_index == 1:
        GPIO.output(green_pin, GPIO.LOW)   # turn off green color
        GPIO.output(red_pin, GPIO.LOW)  # turn off red color
    elif label_index == 2:
        GPIO.output(green_pin, GPIO.HIGH)   # turn on green color
        GPIO.output(red_pin, GPIO.LOW)   # turn off red color
    elif label_index == 3:
        GPIO.output(green_pin, GPIO.LOW)   # turn off green color
        GPIO.output(red_pin, GPIO.HIGH)   # turn on red color


if __name__ == '__main__':
    main()
