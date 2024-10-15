import React, { useEffect, useState } from 'react';
import { Tabs, Tab, Button } from 'react-bootstrap';
import AudioPlayerWithActivation from './AudioPlayerWithActivation';
import Plot from 'react-plotly.js';

const API_BASE_URL = 'http://localhost:5555';  // Replace with your actual IP address

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
  const [selectedFile, setSelectedFile] = useState(null);
  const [localAudioUrl, setLocalAudioUrl] = useState(null);
  const [topN, setTopN] = useState(32);
  const [uploadedFileResults, setUploadedFileResults] = useState(null);

  const [activeKey, setActiveKey] = useState('neuronSearch');

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

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setLocalAudioUrl(URL.createObjectURL(file));
  };

  const handleTopNChange = (event) => {
    setTopN(parseInt(event.target.value, 10));
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload');
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

  const handleTabSelect = (key) => {
    setActiveKey(key);
  };

  return (
    <div className="w-full max-w-3xl mx-auto px-4">
      <h1 className="text-2xl font-bold m-4">{layerName}</h1>
      {error && <p className="text-danger">{error}</p>}

      <Tabs
        activeKey={activeKey}
        onSelect={handleTabSelect}
        className="mb-3"
      >
        <Tab eventKey="neuronSearch" title="Neuron Search">
          {/* Neuron search form */}
          <form onSubmit={handleSubmit} className="mb-4">
            <div className="mb-3">
              <label htmlFor="neuronIdx" className="form-label">Neuron Index:</label>
              <input
                id="neuronIdx"
                type="number"
                value={neuronIdx}
                onChange={handleNeuronChange}
                className="form-control"
                min="0"
                max={nFeatures - 1}
                disabled={isLoading || !isServerReady || !!error}
              />
            </div>
            <div className="mb-3">
              <label htmlFor="maxVal" className="form-label">Max Activation Value (optional):</label>
              <input
                id="maxVal"
                type="number"
                value={maxVal}
                onChange={handleMaxValChange}
                className="form-control"
                step="any"
                disabled={isLoading || !isServerReady || !!error}
              />
            </div>
            <div className="mb-3">
              <label htmlFor="minVal" className="form-label">Min Activation Value (optional):</label>
              <input
                id="minVal"
                type="number"
                value={minVal}
                onChange={handleMinValChange}
                className="form-control"
                step="any"
                disabled={isLoading || !isServerReady || !!error}
              />
            </div>
            <div className="mb-3 form-check">
              <input
                id="useAbs"
                type="checkbox"
                checked={useAbs}
                onChange={handleAbsChange}
                className="form-check-input"
                disabled={isLoading || !isServerReady || !!error}
              />
              <label htmlFor="useAbs" className="form-check-label">Use Absolute Value</label>
            </div>
            <Button
              type="submit"
              disabled={isLoading || !isServerReady || !!error}
            >
              Update
            </Button>
          </form>

          {/* Histogram plot */}
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

          {/* Display for top files */}
          {topFiles.map((file, index) => (
            <AudioPlayerWithActivation
              key={`${file}-${index}`}
              audioFile={file}
              apiBaseUrl={API_BASE_URL}
              activations={activations[index] || []}
            />
          ))}
        </Tab>

        <Tab eventKey="fileUpload" title="File Upload">
          {/* File upload section */}
          <div className="mb-4">
            <h2 className="h4 mb-3">Upload and Analyze Audio</h2>
            <div className="mb-3">
              <input
                type="file"
                onChange={handleFileChange}
                accept="audio/*"
                className="form-control"
              />
            </div>
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
              disabled={!selectedFile || isLoading}
            >
              Upload and Analyze
            </Button>
          </div>

          {/* Display for uploaded file activations */}
          {uploadedFileResults && localAudioUrl && (
            <div>
              <h3 className="h5 mb-3">Top {topN} Activations for Uploaded File</h3>
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
        </Tab>
      </Tabs>

      {isLoading && <p className="text-info">Loading...</p>}
      {!isServerReady && <p className="text-warning">Waiting for server...</p>}
    </div>
  );
};

export default ActivationDisplay;