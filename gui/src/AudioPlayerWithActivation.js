import React, { useEffect, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js';

const API_BASE_URL = 'http://192.168.0.18:5000';  // Replace with your actual IP address

const AudioPlayer = ({ audioFile, neuronIdx }) => {
  const [wavesurfer, setWavesurfer] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activationData, setActivationData] = useState([]);

  useEffect(() => {
    const ws = WaveSurfer.create({
      container: `#waveform-${audioFile.replace(/[\/\.]/g, '-')}`,
      waveColor: 'violet',
      progressColor: 'purple',
      cursorColor: 'navy',
      height: 100,
      responsive: true,
      plugins: [RegionsPlugin.create()]
    });

    ws.load(`${API_BASE_URL}/audio/${encodeURIComponent(audioFile)}`);

    ws.on('ready', () => {
      setWavesurfer(ws);
      fetchActivationData();
    });

    ws.on('play', () => setIsPlaying(true));
    ws.on('pause', () => setIsPlaying(false));

    return () => {
      ws.destroy();
    };
  }, [audioFile]);

  useEffect(() => {
    if (wavesurfer && activationData.length > 0) {
      wavesurfer.regions.clear();

      activationData.forEach((activation, index) => {
        wavesurfer.addRegion({
          start: index / activationData.length * wavesurfer.getDuration(),
          end: (index + 1) / activationData.length * wavesurfer.getDuration(),
          color: getColorFromActivation(activation),
          drag: false,
          resize: false,
        });
      });

      wavesurfer.drawBuffer();
    }
  }, [activationData, wavesurfer]);

  const fetchActivationData = () => {
    fetch(`${API_BASE_URL}/activation?neuron_idx=${neuronIdx}&audio_file=${encodeURIComponent(audioFile)}`)
      .then(response => response.json())
      .then(data => setActivationData(data.activations))
      .catch(error => console.error('Error fetching activation data:', error));
  };

  const togglePlayPause = () => {
    if (wavesurfer) {
      wavesurfer.playPause();
    }
  };

  const getColorFromActivation = (activation) => {
    if (activation === 0) return 'rgba(0, 0, 0, 0)';
    const intensity = Math.abs(activation);
    const alpha = Math.min(intensity * 0.5, 0.5);
    return activation > 0
      ? `rgba(0, 255, 0, ${alpha})`  // Green for positive activations
      : `rgba(255, 0, 0, ${alpha})`; // Red for negative activations
  };

  return (
    <div className="mb-4">
      <div id={`waveform-${audioFile.replace(/[\/\.]/g, '-')}`} />
      <button onClick={togglePlayPause} className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
        {isPlaying ? 'Pause' : 'Play'}
      </button>
      <span className="ml-2">{audioFile}</span>
    </div>
  );
};

const AudioPlayerWithActivation = () => {
  const [neuronIdx, setNeuronIdx] = useState('');
  const [topFiles, setTopFiles] = useState([]);
  const [isServerReady, setIsServerReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/status`)
      .then(response => response.json())
      .then(data => {
        if (data.status === "Initialization complete") {
          setIsServerReady(true);
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

  const handleSubmit = (event) => {
    event.preventDefault();
    if (neuronIdx !== '') {
      const idx = parseInt(neuronIdx, 10);
      if (!isNaN(idx) && idx >= 0) {
        fetchTopFiles(idx);
      } else {
        setError('Please enter a valid non-negative integer');
      }
    }
  };

  const fetchTopFiles = (idx) => {
    setIsLoading(true);
    fetch(`${API_BASE_URL}/top_files?neuron_idx=${idx}&n_files=10`)
      .then(response => response.json())
      .then(data => {
        setTopFiles(data.top_files);
        setIsLoading(false);
      })
      .catch(error => {
        console.error('Error fetching top files:', error);
        setError('Failed to fetch top files');
        setIsLoading(false);
      });
  };

  return (
    <div className="w-full max-w-3xl mx-auto p-4">
      {error && <p className="text-red-500">{error}</p>}
      <form onSubmit={handleSubmit} className="mb-4">
        <input
          type="number"
          value={neuronIdx}
          onChange={handleNeuronChange}
          className="px-2 py-1 border rounded"
          min="0"
          disabled={isLoading || !isServerReady || !!error}
        />
        <button
          type="submit"
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 ml-2"
          disabled={isLoading || !isServerReady || !!error}
        >
          Update
        </button>
      </form>
      {isLoading && <p>Loading...</p>}
      {!isServerReady && <p>Waiting for server...</p>}
      {topFiles.map((file, index) => (
        <AudioPlayer key={index} audioFile={file} neuronIdx={parseInt(neuronIdx, 10)} />
      ))}
    </div>
  );
};

export default AudioPlayerWithActivation;