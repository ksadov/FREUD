import React, { useState } from 'react';
import { Button } from 'react-bootstrap';
import AudioPlayerWithActivation from './AudioPlayerWithActivation';

const AnalyzeAudioTab = ({ API_BASE_URL, selectedFile, localAudioUrl, isLoading, setIsLoading, isRecording, setError }) => {
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

  return (
    <div>
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
        Analyze Audio
      </Button>
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
    </div>
  );
};

export default AnalyzeAudioTab;