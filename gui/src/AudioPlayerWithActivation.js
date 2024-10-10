import React, { useEffect, useState, useCallback, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js';
import SpectrogramPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.spectrogram.min.js';

const API_BASE_URL = 'http://localhost:5555';  // Replace with your actual IP address

const AudioPlayer = ({ audioFile, activations }) => {
  const waveformRef = useRef(null);
  const spectrogramRef = useRef(null);
  const wavesurferRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentActivation, setCurrentActivation] = useState(0);

  const createWavesurfer = useCallback(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy();
    }

    wavesurferRef.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: 'violet',
      progressColor: 'purple',
      cursorColor: 'navy',
      height: 100,
      responsive: true,
      plugins: [
        RegionsPlugin.create(),
        SpectrogramPlugin.create({
          wavesurfer: wavesurferRef.current,
          container: spectrogramRef.current,
          labels: true,
          height: 100,
        })
      ]
    });

    wavesurferRef.current.on('ready', () => {
      setDuration(wavesurferRef.current.getDuration());
      addRegions();
    });

    wavesurferRef.current.on('audioprocess', updateTimeAndActivation);
    wavesurferRef.current.on('seek', updateTimeAndActivation);
    wavesurferRef.current.on('play', () => setIsPlaying(true));
    wavesurferRef.current.on('pause', () => setIsPlaying(false));

    return wavesurferRef.current;
  }, []);

  const updateTimeAndActivation = useCallback(() => {
    if (wavesurferRef.current) {
      const currentTime = wavesurferRef.current.getCurrentTime();
      setCurrentTime(currentTime);

      if (activations.length > 0) {
        const currentTimeRatio = currentTime / wavesurferRef.current.getDuration();
        const activationIndex = Math.floor(currentTimeRatio * activations.length);
        setCurrentActivation(activations[activationIndex] || 0);
      }
    }
  }, [activations]);

  const addRegions = useCallback(() => {
    if (wavesurferRef.current && activations.length > 0) {
      wavesurferRef.current.regions.clear();

      activations.forEach((activation, index) => {
        wavesurferRef.current.addRegion({
          start: index / activations.length * wavesurferRef.current.getDuration(),
          end: (index + 1) / activations.length * wavesurferRef.current.getDuration(),
          color: getColorFromActivation(activation),
          drag: false,
          resize: false,
        });
      });

      wavesurferRef.current.drawBuffer();
    }
  }, [activations]);

  useEffect(() => {
    const wavesurfer = createWavesurfer();
    return () => wavesurfer.destroy();
  }, [createWavesurfer]);

  useEffect(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.load(`${API_BASE_URL}/audio/${encodeURIComponent(audioFile)}`);
    }
  }, [audioFile]);

  const togglePlayPause = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause();
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

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    const milliseconds = Math.floor((time % 1) * 100); // Only two digits
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="mb-4">
      <div ref={waveformRef} />
      <div ref={spectrogramRef} />
      <div className="flex items-center mt-2">
        <button onClick={togglePlayPause} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mr-4">
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <span className="text-sm font-mono mr-4">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        <span className="text-sm font-mono" style={{ color: currentActivation >= 0 ? 'green' : 'red' }}>
          Activation: {currentActivation.toFixed(4)}
        </span>
        <span className="ml-4">{audioFile}</span>
      </div>
    </div>
  );
};

const AudioPlayerWithActivation = () => {
  const [neuronIdx, setNeuronIdx] = useState('');
  const [maxVal, setMaxVal] = useState('');
  const [topFiles, setTopFiles] = useState([]);
  const [activations, setActivations] = useState([]);
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

  const handleMaxValChange = (event) => {
    setMaxVal(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (neuronIdx !== '') {
      const idx = parseInt(neuronIdx, 10);
      if (!isNaN(idx) && idx >= 0) {
        fetchTopFiles(idx);
      } else {
        setError('Please enter a valid non-negative integer for neuron index');
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
    fetch(url)
      .then(response => response.json())
      .then(data => {
        setTopFiles(data.top_files);
        setActivations(data.activations);
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
        <div className="mb-2">
          <label htmlFor="neuronIdx" className="mr-2">Neuron Index:</label>
          <input
            id="neuronIdx"
            type="number"
            value={neuronIdx}
            onChange={handleNeuronChange}
            className="px-2 py-1 border rounded"
            min="0"
            disabled={isLoading || !isServerReady || !!error}
          />
        </div>
        <div className="mb-2">
          <label htmlFor="maxVal" className="mr-2">Max Activation Value (optional):</label>
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
        <button
          type="submit"
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          disabled={isLoading || !isServerReady || !!error}
        >
          Update
        </button>
      </form>
      {isLoading && <p>Loading...</p>}
      {!isServerReady && <p>Waiting for server...</p>}
      {topFiles.map((file, index) => (
        <AudioPlayer key={`${file}-${index}`} audioFile={file} activations={activations[index] || []} />
      ))}
    </div>
  );
};

export default AudioPlayerWithActivation;