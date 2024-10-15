import React, { useState, useRef } from 'react';
import { Button } from 'react-bootstrap';
import encodeWAV from 'audiobuffer-to-wav';

const AudioRecorder = ({ onRecordingStart, onRecordingComplete, disabled }) => {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        convertToWav(audioBlob).then(wavBlob => {
          onRecordingComplete(wavBlob);
        });
        audioChunksRef.current = [];
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      onRecordingStart();
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const convertToWav = (blob) => {
    return new Promise((resolve) => {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const fileReader = new FileReader();

      fileReader.onload = (e) => {
        const arrayBuffer = e.target.result;
        audioContext.decodeAudioData(arrayBuffer, (audioBuffer) => {
          const wavBlob = encodeWAV(audioBuffer);
          resolve(new Blob([wavBlob], { type: 'audio/wav' }));
        });
      };

      fileReader.readAsArrayBuffer(blob);
    });
  };

  return (
    <div>
      {!isRecording ? (
        <Button onClick={startRecording} disabled={disabled}>Start Recording</Button>
      ) : (
        <Button onClick={stopRecording} variant="danger">Stop Recording</Button>
      )}
    </div>
  );
};

export default AudioRecorder;