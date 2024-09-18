import React from 'react';
import AudioPlayerWithActivation from './AudioPlayerWithActivation';

function App() {
  const audioFilename = '5764-299665-0090.flac';  // Update this to the correct filename

  return (
    <div className="App">
      <h1>Whisper Model Visualization</h1>
      <AudioPlayerWithActivation audioFilename={audioFilename} />
    </div>
  );
}

export default App;