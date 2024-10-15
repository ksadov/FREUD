import React, { useState, useRef } from 'react';
import { Button, Nav, Tab } from 'react-bootstrap';
import AudioRecorder from './AudioRecorder';
import AnalyzeAudioTab from './AnalyzeAudioTab';
import ManipulateFeatureTab from './ManipulateFeatureTab';

const AudioUploadTab = ({ API_BASE_URL }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [localAudioUrl, setLocalAudioUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [hasAudio, setHasAudio] = useState(false);
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setLocalAudioUrl(URL.createObjectURL(file));
      setHasAudio(true);
    }
  };

  const handleRecordingStart = () => {
    setIsRecording(true);
  };

  const handleRecordingComplete = (blob) => {
    setSelectedFile(new File([blob], "recorded_audio.wav", { type: "audio/wav" }));
    setLocalAudioUrl(URL.createObjectURL(blob));
    setIsRecording(false);
    setHasAudio(true);
  };

  const handleDiscardAudio = () => {
    if (localAudioUrl) {
      URL.revokeObjectURL(localAudioUrl);
    }
    setSelectedFile(null);
    setLocalAudioUrl(null);
    setHasAudio(false);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div>
      <h2 className="h4 mb-3">Upload or Record and Analyze Audio</h2>
      <div className="mb-3">
        <input
          ref={fileInputRef}
          type="file"
          onChange={handleFileChange}
          accept="audio/*"
          className="form-control"
          disabled={isRecording || hasAudio}
        />
      </div>
      <div className="mb-3">
        <h3 className="h5">Or Record Audio</h3>
        <AudioRecorder
          onRecordingStart={handleRecordingStart}
          onRecordingComplete={handleRecordingComplete}
          disabled={hasAudio}
        />
      </div>
      {localAudioUrl && (
        <div className="mb-3">
          <h3 className="h5">Preview Recorded/Uploaded Audio</h3>
          <div className="d-flex align-items-center">
            <audio ref={audioRef} controls src={localAudioUrl} className="flex-grow-1 me-2" />
            <Button variant="danger" onClick={handleDiscardAudio}>Discard</Button>
          </div>
        </div>
      )}
      <div className="border border-2 rounded p-2">
        <Tab.Container id="analysis-tabs" defaultActiveKey="analyze">
          <Nav variant="tabs" className="mb-3">
            <Nav.Item>
              <Nav.Link eventKey="analyze">Analyze Audio</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="manipulate">Manipulate Feature</Nav.Link>
            </Nav.Item>
          </Nav>
          <Tab.Content>
            <Tab.Pane eventKey="analyze">
              <AnalyzeAudioTab
                API_BASE_URL={API_BASE_URL}
                selectedFile={selectedFile}
                localAudioUrl={localAudioUrl}
                isLoading={isLoading}
                setIsLoading={setIsLoading}
                isRecording={isRecording}
                setError={setError}
              />
            </Tab.Pane>
            <Tab.Pane eventKey="manipulate">
              <ManipulateFeatureTab
                API_BASE_URL={API_BASE_URL}
                selectedFile={selectedFile}
                localAudioUrl={localAudioUrl}
                isLoading={isLoading}
                setIsLoading={setIsLoading}
                isRecording={isRecording}
                setError={setError}
              />
            </Tab.Pane>
          </Tab.Content>
        </Tab.Container>
      </div>

      {isLoading && <p className="text-info">Loading...</p>}
      {error && <p className="text-danger">{error}</p>}
    </div>
  );
};

export default AudioUploadTab;