import React, { useEffect, useState } from 'react';
import Button from 'react-bootstrap/Button';
import AudioPlayerWithActivation from './AudioPlayerWithActivation';
import Plot from 'react-plotly.js';

const API_BASE_URL = 'http://192.168.0.17:5555';  // Replace with your actual IP address

const ActivationDisplay = () => {
  const [neuronIdx, setNeuronIdx] = useState('');
  const [maxVal, setMaxVal] = useState('');
  const [minVal, setMinVal] = useState('');
  const [useAbs, setUseAbs] = useState(false);
  const [topFiles, setTopFiles] = useState([]);
  const [activations, setActivations] = useState([]);
  const [maxPerFile, setMaxPerFile] = useState([]);
  const [isServerReady, setIsServerReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [layerName, setLayerName] = useState('');
  const [nFeatures, setNFeatures] = useState(0);

  useEffect(() => {
    fetch(`${API_BASE_URL}/status`)
      .then(response => response.json())
      .then(data => {
        if (data.status === "Initialization complete") {
          setIsServerReady(true);
          setLayerName(data.layer_name);
          setNFeatures(data.n_features);
        } else {
          setError('Server not ready');
        }
      })
      .catch(error => {
        console.error('Error checking server status:', error);
        setError('Failed to connect to server');
      });
  }, []);

  const handleNeuronChange = (event) => {
    setNeuronIdx(event.target.value);
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
    if (neuronIdx !== '') {
      const idx = parseInt(neuronIdx, 10);
      if (!isNaN(idx) && idx >= 0 && idx < nFeatures) {
        fetchTopFiles(idx);
      } else {
        setError(`Please enter a valid neuron index between 0 and ${nFeatures - 1}`);
      }
    }
  };

  const fetchTopFiles = (idx) => {
    setIsLoading(true);
    let url = `${API_BASE_URL}/top_files?neuron_idx=${idx}&n_files=20`;
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
    <div className="w-full max-w-3xl mx-auto px-4">
      <h1 className="text-2xl font-bold m-4">{layerName}</h1>
      {error && <p className="text-red-500">{error}</p>}
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="mb-2">
          <label htmlFor="neuronIdx" className="mx-2">Neuron Index:</label>
          <input
            id="neuronIdx"
            type="number"
            value={neuronIdx}
            onChange={handleNeuronChange}
            className="px-2 py-1 border rounded"
            min="0"
            max={nFeatures - 1}
            disabled={isLoading || !isServerReady || !!error}
          />
        </div>
        <div className="mb-2">
          <label htmlFor="maxVal" className="mx-2">Max Activation Value (optional):</label>
          <input
            id="maxVal"
            type="number"
            value={maxVal}
            onChange={handleMaxValChange}
            className="px-2 py-1 border rounded"
            step="any"
            disabled={isLoading || !isServerReady || !!error}
          />
        </div>
        <div className="mb-2">
          <label htmlFor="minVal" className="mx-2">Min Activation Value (optional):</label>
          <input
            id="minVal"
            type="number"
            value={minVal}
            onChange={handleMinValChange}
            className="px-2 py-1 border rounded"
            step="any"
            disabled={isLoading || !isServerReady || !!error}
          />
        </div>
        <div className="mb-2">
          <label htmlFor="useAbs" className="mx-2">Use Absolute Value:</label>
          <input
            id="useAbs"
            type="checkbox"
            checked={useAbs}
            onChange={handleAbsChange}
            disabled={isLoading || !isServerReady || !!error}
          />
        </div>
        <Button
          type="submit"
          className="px-4 py-2"
          disabled={isLoading || !isServerReady || !!error}
        >
          Update
        </Button>
      </form>
      {isLoading && <p>Loading...</p>}
      {!isServerReady && <p>Waiting for server...</p>}
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
        <AudioPlayerWithActivation key={`${file}-${index}`} audioFile={file} apiBaseUrl={API_BASE_URL} activations={activations[index] || []} />
      ))}
    </div>
  );
};

export default ActivationDisplay;