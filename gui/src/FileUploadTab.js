import React, { useState, useRef } from 'react';
import { Button, Form } from 'react-bootstrap';
import AudioPlayerWithActivation from './AudioPlayerWithActivation';
import AudioRecorder from './AudioRecorder';

const FileUploadTab = ({ API_BASE_URL }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [localAudioUrl, setLocalAudioUrl] = useState(null);
  const [topN, setTopN] = useState(32);
  const [uploadedFileResults, setUploadedFileResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [hasAudio, setHasAudio] = useState(false);
  const [featureIndex, setFeatureIndex] = useState(0);
  const [ablationFactor, setAblationFactor] = useState(1.5);
  const [manipulationResults, setManipulationResults] = useState(null);
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

  const handleTopNChange = (event) => {
    setTopN(parseInt(event.target.value, 10));
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
    setUploadedFileResults(null);
    setManipulationResults(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload or record audio');
      return;
    }
    setIsLoading(true);
    const formData = new FormData();
    formData.append('audio', selectedFile);
    try {
      const response = await fetch(`${API_BASE_URL}/analyze_audio?top_n=${topN}`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      console.log('Uploaded file results:', data);
      setUploadedFileResults(data);
      setIsLoading(false);
    } catch (error) {
      console.error('Error uploading and analyzing file:', error);
      setError('Failed to upload and analyze file');
      setIsLoading(false);
    }
  };

  const handleManipulateFeature = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Please select a file to upload or record audio');
      return;
    }
    setIsLoading(true);
    const formData = new FormData();
    formData.append('audio', selectedFile);
    try {
      const response = await fetch(`${API_BASE_URL}/manipulate_feature?feat_idx=${featureIndex}&manipulation_factor=${ablationFactor}`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      console.log('Manipulation results:', data);
      setManipulationResults(data);
      setIsLoading(false);
    } catch (error) {
      console.error('Error manipulating feature:', error);
      setError('Failed to manipulate feature');
      setIsLoading(false);
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
      <div className="mb-3">
        <label htmlFor="topN" className="form-label">Top N Activations:</label>
        <input
          id="topN"
          type="number"
          value={topN}
          onChange={handleTopNChange}
          className="form-control"
          min="1"
        />
      </div>
      <Button
        onClick={handleFileUpload}
        disabled={!selectedFile || isLoading || isRecording}
        className="mb-3"
      >
        Upload and Analyze
      </Button>

      <Form onSubmit={handleManipulateFeature} className="mb-3">
        <Form.Group className="mb-3">
          <Form.Label>Feature Index</Form.Label>
          <Form.Control
            type="number"
            value={featureIndex}
            onChange={(e) => setFeatureIndex(parseInt(e.target.value))}
            min="0"
          />
        </Form.Group>
        <Form.Group className="mb-3">
          <Form.Label>Ablation Factor</Form.Label>
          <Form.Control
            type="number"
            value={ablationFactor}
            onChange={(e) => setAblationFactor(parseFloat(e.target.value))}
            step="0.1"
          />
        </Form.Group>
        <Button type="submit" disabled={!selectedFile || isLoading || isRecording}>
          Manipulate Feature
        </Button>
      </Form>

      {isLoading && <p className="text-info">Loading...</p>}
      {error && <p className="text-danger">{error}</p>}

      {uploadedFileResults && localAudioUrl && (
        <div>
          <h3 className="h5 my-3">Top {topN} Activations for Uploaded/Recorded File</h3>
          {uploadedFileResults.top_indices.map((neuronIndex, idx) => (
            <div key={neuronIndex} className="mb-4">
              <h4 className="h6">Neuron {neuronIndex}</h4>
              <AudioPlayerWithActivation
                audioFile={localAudioUrl}
                activations={uploadedFileResults.top_activations[idx]}
                isLocalFile={true}
                neuronIndex={neuronIndex}
              />
            </div>
          ))}
        </div>
      )}

      {manipulationResults && localAudioUrl && (
        <div>
          <h3 className="h5 my-3">Feature Manipulation Results</h3>
          {manipulationResults.baseline_text && (
            <div className="mb-3">
              <h4 className="h6">Baseline Text</h4>
              <p>{manipulationResults.baseline_text}</p>
            </div>
          )}
          <div className="mb-4">
            <h4 className="h6">Standard Activations</h4>
            <AudioPlayerWithActivation
              audioFile={localAudioUrl}
              activations={manipulationResults.standard_activations}
              isLocalFile={true}
              neuronIndex={featureIndex}
            />
            <p><strong>Standard Text:</strong> {manipulationResults.standard_text}</p>
          </div>
          <div className="mb-4">
            <h4 className="h6">Manipulated Activations</h4>
            <AudioPlayerWithActivation
              audioFile={localAudioUrl}
              activations={manipulationResults.manipulated_activations}
              isLocalFile={true}
              neuronIndex={featureIndex}
            />
            <p><strong>Manipulated Text:</strong> {manipulationResults.manipulated_text}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUploadTab;