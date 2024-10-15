import React, { useState } from 'react';
import { Button, Form } from 'react-bootstrap';
import AudioPlayerWithActivation from './AudioPlayerWithActivation';
import Plot from 'react-plotly.js';

const ActivationSearchTab = ({ isServerReady, nFeatures, API_BASE_URL }) => {
  const [featureIdx, setFeatureIdx] = useState('');
  const [nResults, setNResults] = useState(20);
  const [maxVal, setMaxVal] = useState('');
  const [minVal, setMinVal] = useState('');
  const [useAbs, setUseAbs] = useState(false);
  const [topFiles, setTopFiles] = useState([]);
  const [activations, setActivations] = useState([]);
  const [maxPerFile, setMaxPerFile] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFeatureChange = (event) => {
    setFeatureIdx(event.target.value);
  };

  const handleNResultsChange = (event) => {
    setNResults(event.target.value);
  };

  const handleMaxValChange = (event) => {
    setMaxVal(event.target.value);
  };

  const handleMinValChange = (event) => {
    setMinVal(event.target.value);
  };

  const handleAbsChange = (event) => {
    setUseAbs(event.target.checked);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (featureIdx !== '') {
      const idx = parseInt(featureIdx, 10);
      if (!isNaN(idx) && idx >= 0 && idx < nFeatures) {
        fetchTopFiles(idx, nResults);
      } else {
        setError(`Please enter a valid feature index between 0 and ${nFeatures - 1}`);
      }
    }
  };

  const fetchTopFiles = (idx, nResults) => {
    setIsLoading(true);
    let url = `${API_BASE_URL}/top_files?feature_idx=${idx}&n_files=${nResults}`;
    if (maxVal !== '') {
      const maxValFloat = parseFloat(maxVal);
      if (!isNaN(maxValFloat)) {
        url += `&max_val=${maxValFloat}`;
      } else {
        setError('Please enter a valid number for max value');
        setIsLoading(false);
        return;
      }
    }
    if (minVal !== '') {
      const minValFloat = parseFloat(minVal);
      if (!isNaN(minValFloat)) {
        url += `&min_val=${minValFloat}`;
      } else {
        setError('Please enter a valid number for min value');
        setIsLoading(false);
        return;
      }
    }
    if (useAbs) {
      url += '&absolute_magnitude=true';
    }
    fetch(url)
      .then(response => response.json())
      .then(data => {
        setTopFiles(data.top_files);
        setActivations(data.activations);
        setMaxPerFile(data.max_per_file);
        setIsLoading(false);
      })
      .catch(error => {
        console.error('Error fetching top files:', error);
        setError('Failed to fetch top files');
        setIsLoading(false);
      });
  };

  return (
    <>
      <Form onSubmit={handleSubmit} className="mb-4">
        <Form.Group>
          <Form.Label htmlFor="featureIdx">Feature Index:</Form.Label>
          <Form.Control
            id="featureIdx"
            type="number"
            value={featureIdx}
            onChange={handleFeatureChange}
            min="0"
            max={nFeatures - 1}
            disabled={isLoading || !isServerReady || !!error}
          />
        </Form.Group>
        <Form.Group>
          <Form.Label htmlFor="nResults">Max Number of Results:</Form.Label>
          <Form.Control
            id="nResults"
            type="number"
            value={nResults}
            onChange={handleNResultsChange}
            min="1"
            disabled={isLoading || !isServerReady || !!error}
          />
        </Form.Group>
        <Form.Group>
          <Form.Label htmlFor="maxVal">Max Activation Value (optional):</Form.Label>
          <Form.Control
            id="maxVal"
            type="number"
            value={maxVal}
            onChange={handleMaxValChange}
            step="any"
            disabled={isLoading || !isServerReady || !!error}
          />
        </Form.Group>
        <Form.Group>
          <Form.Label htmlFor="minVal">Min Activation Value (optional):</Form.Label>
          <Form.Control
            id="minVal"
            type="number"
            value={minVal}
            onChange={handleMinValChange}
            step="any"
            disabled={isLoading || !isServerReady || !!error}
          />
        </Form.Group>
        <Form.Check>
          <Form.Check
            id="useAbs"
            type="checkbox"
            checked={useAbs}
            onChange={handleAbsChange}
            disabled={isLoading || !isServerReady || !!error}
          />
          <Form.Label htmlFor="useAbs">Use Absolute Value</Form.Label>
        </Form.Check>
        <Button
          type="submit"
          disabled={isLoading || !isServerReady || !!error}
        >
          Update
        </Button>
      </Form>

      {isLoading && <p className="text-info">Loading...</p>}

      {maxPerFile.length > 0 && (
        <Plot
          data={[
            {
              x: maxPerFile,
              type: 'histogram',
              marker: {
                color: 'rgba(0,0,255,0.7)',
              },
            },
          ]}
          layout={{
            width: 720,
            height: 480,
            title: 'Histogram of Max Activation Values per File',
            xaxis: { title: 'Max Activation Value' },
            yaxis: { title: 'Count', type: 'log', autorange: true },
          }}
        />
      )}

      {topFiles.map((file, index) => (
        <AudioPlayerWithActivation
          key={`${file}-${index}`}
          audioFile={file}
          apiBaseUrl={API_BASE_URL}
          activations={activations[index] || []}
        />
      ))}
    </>
  );
};

export default ActivationSearchTab;