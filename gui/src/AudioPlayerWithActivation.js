import React, { useEffect, useState, useCallback, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions.min.js';
import SpectrogramPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.spectrogram.min.js';
import CursorPlugin from 'wavesurfer.js/src/plugin/cursor';
import Button from 'react-bootstrap/Button';
import { IoMdPlay, IoMdPause } from "react-icons/io";

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
      waveColor: 'rgba(0,0,0,0)',
      progressColor: 'rgba(0,0,0,0)',
      cursorColor: 'navy',
      height: 100,
      responsive: true,
      plugins: [
        RegionsPlugin.create(),
        SpectrogramPlugin.create({
          wavesurfer: wavesurferRef.current,
          container: spectrogramRef.current,
          // Ensure spectrogram height matches the waveform height for proper overlay
          height: 100,
          scale: 'mel',
          fftSamples: 512,
          windowFunc: 'hann'
        }),
        CursorPlugin.create({
          showTime: true,
          opacity: 1,
          customShowTimeStyle: {
            'background-color': '#000',
            color: '#fff',
            padding: '2px',
            'font-size': '10px'
          }
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
    wavesurferRef.current.on('redraw', () => wavesurferRef.current.spectrogram.init());

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

      const maxActivation = Math.max(...activations.map(Math.abs));

      activations.forEach((activation, index) => {
        wavesurferRef.current.addRegion({
          start: index / activations.length * wavesurferRef.current.getDuration(),
          end: (index + 1) / activations.length * wavesurferRef.current.getDuration(),
          color: getColorFromActivation(activation / maxActivation),
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
    <div className="mb-4 border rounded border-2 ">
      {/* Container for both waveform and spectrogram to overlay them */}
      <div style={{ position: 'relative', height: '100px' }} alt={audioFile}>
        {/* Waveform container */}
        <div ref={waveformRef} style={{ position: 'absolute', width: '100%', height: '100%' }} />
        {/* Spectrogram container overlaying the waveform */}
        <div ref={spectrogramRef} style={{ position: 'absolute', width: '100%', height: '100%', zIndex: -100 }} />
      </div>
      <div className="flex bg-light items-center p-1">
        <Button onClick={togglePlayPause} variant="outline-primary" className=" px-2 py-1">
          {isPlaying ? <IoMdPause size={24} /> : <IoMdPlay size={24} />}
        </Button>
        <span className="text-sm font-mono mx-2">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        <span className="text-sm font-mono" style={{ color: currentActivation >= 0 ? 'green' : 'red' }}>
          Activation: {currentActivation.toFixed(4)}
        </span>
      </div>
    </div >
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
    <div className="w-full max-w-3xl mx-auto px-4">
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
      {topFiles.map((file, index) => (
        <AudioPlayer key={`${file}-${index}`} audioFile={file} activations={activations[index] || []} />
      ))}
    </div>
  );
};

export default AudioPlayerWithActivation;