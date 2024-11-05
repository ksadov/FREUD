import React, { useEffect, useState } from 'react';
import { Tabs, Tab } from 'react-bootstrap';
import ActivationSearchTab from './ActivationSearchTab';
import UploadAudioTab from './UploadAudioTab';

const API_BASE_URL = 'http://localhost:5555';  // Replace with your actual IP address

const ActivationDisplay = () => {
  const [isServerReady, setIsServerReady] = useState(false);
  const [error, setError] = useState(null);
  const [layerName, setLayerName] = useState('');
  const [nFeatures, setNFeatures] = useState(0);
  const [activeKey, setActiveKey] = useState('activationSearch');
  const [allowAudioUpload, setAllowAudioUpload] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE_URL}/status`)
      .then(response => response.json())
      .then(data => {
        if (data.status === "Initialization complete") {
          setIsServerReady(true);
          setLayerName(data.layer_name);
          setNFeatures(data.n_features);
          setAllowAudioUpload(data.allow_audio_upload ?? true); // Default to true if not specified
        } else {
          setError('Server not ready');
        }
      })
      .catch(error => {
        console.error('Error checking server status:', error);
        setError('Failed to connect to server');
      });
  }, []);

  const handleTabSelect = (key) => {
    setActiveKey(key);
  };

  // If audio upload is disabled and that tab is selected, switch to activation search
  useEffect(() => {
    if (!allowAudioUpload && activeKey === 'uploadaudio') {
      setActiveKey('activationSearch');
    }
  }, [allowAudioUpload, activeKey]);

  return (
    <div className="w-full max-w-3xl mx-auto px-4">
      <h1 className="text-2xl font-bold m-4">{layerName}</h1>
      {error && <p className="text-danger">{error}</p>}
      <Tabs
        activeKey={activeKey}
        onSelect={handleTabSelect}
        className="mb-3"
      >
        <Tab eventKey="activationSearch" title="Activation Search">
          <ActivationSearchTab
            isServerReady={isServerReady}
            nFeatures={nFeatures}
            API_BASE_URL={API_BASE_URL}
          />
        </Tab>
        {allowAudioUpload && (
          <Tab eventKey="uploadaudio" title="Upload Audio">
            <UploadAudioTab
              API_BASE_URL={API_BASE_URL}
            />
          </Tab>
        )}
      </Tabs>
      {!isServerReady && <p className="text-warning">Waiting for server...</p>}
    </div>
  );
};

export default ActivationDisplay;