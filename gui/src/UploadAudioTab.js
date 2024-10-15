import React, { useState, useRef } from 'react';
import { Button, Nav, Tab, Row, Col } from 'react-bootstrap';
import AudioRecorder from './AudioRecorder';
import TopFeaturesTab from './TopFeaturesTab';
import ManipulateFeatureTab from './ManipulateFeatureTab';

const UploadAudioTab = ({ API_BASE_URL }) => {
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
      <Row className="mb-3">
        <Row className="mb-2">
          <h3>Upload or Record Audio</h3>
        </Row>
        <Row>
          <Col>
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              accept="audio/*"
              className="form-control"
              disabled={isRecording || hasAudio}
            />
          </Col>
          <Col>
            <AudioRecorder
              onRecordingStart={handleRecordingStart}
              onRecordingComplete={handleRecordingComplete}
              disabled={hasAudio}
            />
          </Col>
        </Row>
      </Row>
      <Row>
        {localAudioUrl && (
          <div className="mb-3">
            <Row className="mb-2">
              <h5>Preview Recorded/Uploaded Audio</h5>
            </Row>
            <Row className="align-items-center">
              <Col className="d-flex">
                <audio ref={audioRef} controls src={localAudioUrl} className="flex-grow-1 me-2" />
              </Col>
              <Col>
                <Button variant="danger" onClick={handleDiscardAudio}>Discard</Button>
              </Col>
            </Row>
          </div>
        )}
      </Row>
      <Row className="border border-2 rounded p-2">
        <Tab.Container id="analysis-tabs" defaultActiveKey="topfeatures">
          <Nav variant="tabs" className="mb-3">
            <Nav.Item>
              <Nav.Link eventKey="topfeatures">Top Features</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="manipulate">Manipulate Feature</Nav.Link>
            </Nav.Item>
          </Nav>
          <Tab.Content>
            <Tab.Pane eventKey="topfeatures">
              <TopFeaturesTab
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
      </Row>

      {isLoading && <p className="text-info">Loading...</p>}
      {error && <p className="text-danger">{error}</p>}
    </div>
  );
};

export default UploadAudioTab;