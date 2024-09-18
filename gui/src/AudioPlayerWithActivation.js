import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js';

const API_BASE_URL = 'http://192.168.0.18:5000';  // Replace with your actual IP address

const AudioPlayerWithActivation = ({ audioFilename }) => {
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activationData, setActivationData] = useState([]);
  const [neuronIdx, setNeuronIdx] = useState('');
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Initialize the server
    fetch(`${API_BASE_URL}/init`)
      .then(response => response.json())
      .then(data => {
        console.log(data);
        setIsInitialized(true);
      })
      .catch(error => {
        console.error('Error initializing server:', error);
        setError('Failed to initialize server');
      });

    // Initialize WaveSurfer
    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: 'violet',
      progressColor: 'purple',
      cursorColor: 'navy',
      height: 100,
      responsive: true,
      plugins: [
        RegionsPlugin.create()
      ]
    });

    const audioUrl = `${API_BASE_URL}/audio/${audioFilename}`;
    console.log('Loading audio from:', audioUrl);

    wavesurfer.current.load(audioUrl);

    wavesurfer.current.on('ready', () => {
      console.log('WaveSurfer is ready');
      wavesurfer.current.drawBuffer();
    });

    wavesurfer.current.on('error', (e) => {
      console.error('WaveSurfer error:', e);
      setError(`Failed to load audio: ${e.message}`);
    });

    wavesurfer.current.on('play', () => setIsPlaying(true));
    wavesurfer.current.on('pause', () => setIsPlaying(false));

    // Ensure proper cleanup
    return () => {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
      }
    };
  }, [audioFilename]);

  useEffect(() => {
    if (isInitialized && neuronIdx !== '') {
      setIsLoading(true);
      // Fetch activation data when neuronIdx changes and component is initialized
      fetch(`${API_BASE_URL}/activation?neuron_idx=${neuronIdx}`)
        .then(response => response.json())
        .then(data => {
          setActivationData(data.activations);
          setIsLoading(false);
        })
        .catch(error => {
          console.error('Error:', error);
          setIsLoading(false);
          setError('Failed to fetch activation data');
        });
    }
  }, [neuronIdx, isInitialized]);

  useEffect(() => {
    if (wavesurfer.current && activationData.length > 0) {
      // Clear existing regions
      wavesurfer.current.regions.clear();

      // Add new regions based on activationData
      activationData.forEach((activation, index) => {
        wavesurfer.current.addRegion({
          start: index / activationData.length * wavesurfer.current.getDuration(),
          end: (index + 1) / activationData.length * wavesurfer.current.getDuration(),
          color: getColorFromActivation(activation),
          drag: false,
          resize: false,
        });
      });

      // Force a redraw of the waveform
      wavesurfer.current.drawBuffer();
    }
  }, [activationData]);

  const togglePlayPause = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
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

  const handleNeuronChange = (event) => {
    const value = event.target.value;
    setNeuronIdx(value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (neuronIdx !== '') {
      // Trigger the effect to fetch new activation data
      setNeuronIdx(neuronIdx);
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto p-4">
      <div ref={waveformRef} className="mb-4" />
      {error && <p className="text-red-500">{error}</p>}
      <button
        onClick={togglePlayPause}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mr-4"
        disabled={!isInitialized || !!error}
      >
        {isPlaying ? 'Pause' : 'Play'}
      </button>
      <form onSubmit={handleSubmit} className="inline">
        <input
          type="number"
          value={neuronIdx}
          onChange={handleNeuronChange}
          className="px-2 py-1 border rounded"
          min="0"
          disabled={isLoading || !isInitialized || !!error}
        />
        <button
          type="submit"
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 ml-2"
          disabled={isLoading || !isInitialized || !!error}
        >
          Update
        </button>
      </form>
      {isLoading && <span className="ml-2">Loading...</span>}
      {!isInitialized && <p>Initializing...</p>}
    </div>
  );
};

export default AudioPlayerWithActivation;