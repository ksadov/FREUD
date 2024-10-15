import React, { useState } from 'react';
import { Button, Form } from 'react-bootstrap';
import AudioPlayerWithActivation from './AudioPlayerWithActivation';

const ManipulateFeatureTab = ({ API_BASE_URL, selectedFile, localAudioUrl, isLoading, setIsLoading, isRecording, setError }) => {
  const [featureIndex, setFeatureIndex] = useState(0);
  const [ablationFactor, setAblationFactor] = useState(1.5);
  const [manipulationResults, setManipulationResults] = useState(null);

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

export default ManipulateFeatureTab;