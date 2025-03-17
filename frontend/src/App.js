import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import ReactPlayer from 'react-player';
import axios from 'axios';
import './App.css';

function App() {
  const [video, setVideo] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const analyzeVideo = useCallback(async (file) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:8000/api/analyze-pitch', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setAnalysis(response.data);
    } catch (err) {
      console.error('Analysis error details:', err.response || err.message);
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError('Unable to analyze pitch. Please try again later.');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file && file.type.startsWith('video/')) {
        setVideo(URL.createObjectURL(file));
        setAnalysis(null);
        setError(null);
        analyzeVideo(file);
      } else {
        setError('Please upload a valid video file (MP4, MOV, or AVI)');
      }
    }
  }, [analyzeVideo]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi']
    },
    maxFiles: 1
  });

  const renderAnalysisResults = useCallback(() => {
    if (!analysis) return null;

    return (
      <div className="mt-8 space-y-4">
        <h2 className="text-2xl font-semibold">Analysis Results</h2>
        
        {/* Metrics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="font-medium">Stride Length</h3>
            <p className="text-2xl">{analysis.summary.average_stride_length.toFixed(2)}</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="font-medium">Arm Angle</h3>
            <p className="text-2xl">{analysis.summary.average_arm_angle.toFixed(2)}°</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="font-medium">Hip Rotation</h3>
            <p className="text-2xl">{analysis.summary.average_hip_angle.toFixed(2)}°</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="font-medium">Knee Flexion</h3>
            <p className="text-2xl">{analysis.summary.average_knee_angle.toFixed(2)}°</p>
          </div>
        </div>

        {/* Feedback */}
        <div className="mt-6">
          <h3 className="font-medium mb-2">Feedback</h3>
          <ul className="space-y-2">
            {analysis.summary.common_feedback.map((item, index) => (
              <li key={index} className="flex items-center">
                <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  }, [analysis]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Pitching Mechanics Analyzer</h1>
      </header>
      <main>
        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the video here ...</p>
          ) : (
            <p>Drag and drop a video file here, or click to select one</p>
          )}
        </div>

        {error && (
          <div className="error p-4 bg-red-100 text-red-700 rounded-lg mt-4">
            {error}
          </div>
        )}
        
        {isLoading && (
          <div className="loading p-4 bg-blue-100 text-blue-700 rounded-lg mt-4">
            Analyzing video...
          </div>
        )}

        {video && (
          <div className="video-preview mt-4">
            <h2>Video Preview</h2>
            <video controls src={video} className="max-w-full rounded-lg shadow-lg" />
          </div>
        )}

        {analysis && (
          <div className="analysis-results mt-8">
            <h2>Analysis Results</h2>
            {renderAnalysisResults()}
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 