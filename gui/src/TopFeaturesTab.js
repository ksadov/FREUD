import React, { useState } from 'react';
import { Button, Row, Col, Form } from 'react-bootstrap';
import AudioPlayerWithActivation from './AudioPlayerWithActivation';

const TopFeaturesTab = ({ API_BASE_URL, selectedFile, localAudioUrl, isLoading, setIsLoading, isRecording, setError }) => {
  const [topN, setTopN] = useState(32);
  const [uploadedFileResults, setUploadedFileResults] = useState(null);

  const handleTopNChange = (event) => {
    setTopN(parseInt(event.target.value, 10));
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
      const response = await fetch(`${API_BASE_URL}/top_features?top_n=${topN}`, {
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
      console.error('Error getting top features for file:', error);
      setError('Failed to get top features for file');
      setIsLoading(false);
    }
  };

  return (
    <div>
      <Row className="mb-3">
        <Col sm="auto">
          <Form.Group>
            <Form.Label htmlFor="topN" className="form-label">Max Number of Results:</Form.Label>
            <Form.Control
              id="topN"
              type="number"
              value={topN}
              onChange={handleTopNChange}
              className="form-control"
              min="1"
            />
          </Form.Group>
        </Col>
      </Row>
      <Button
        onClick={handleFileUpload}
        disabled={!selectedFile || isLoading || isRecording}
        className="mb-3"
      >
        Get Top Features
      </Button>
      {uploadedFileResults && localAudioUrl && (
        <div>
          <h3 className="h5 my-3">Top {topN} Activations for Uploaded/Recorded File</h3>
          {uploadedFileResults.top_indices.map((featureIndex, idx) => (
            <div key={featureIndex} className="mb-4">
              <h4 className="h6">feature {featureIndex}</h4>
              <AudioPlayerWithActivation
                audioFile={localAudioUrl}
                activations={uploadedFileResults.top_activations[idx]}
                isLocalFile={true}
                featureIndex={featureIndex}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TopFeaturesTab;