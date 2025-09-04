import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { useLocation } from 'react-router-dom';

const Container = styled.div`
  min-height: 100vh;
  padding: 80px 2rem 2rem;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: white;
`;

const Content = styled.div`
  max-width: 1200px;
  margin: 0 auto;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 3rem;
`;

const Title = styled.h1`
  font-size: 3rem;
  font-weight: 800;
  margin-bottom: 1rem;
  font-family: 'Inter', sans-serif;
  background: linear-gradient(135deg, #ff6ac1, #38bdf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  opacity: 0.9;
  font-family: 'Inter', sans-serif;
`;

const ImageGrid = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 2rem;
  margin-top: 2rem;
  max-width: 1200px;
  margin-left: auto;
  margin-right: auto;
`;

const ImageContainer = styled.div`
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  max-width: 350px;
`;

const ImageCaption = styled.div`
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(10px);
  border-radius: 10px;
  padding: 1.5rem;
  margin-top: 1rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  font-family: 'Inter', sans-serif;
  font-size: 1.1rem;
  line-height: 1.5;
  color: #e2e8f0;
  text-align: center;
  max-width: 100%;
  word-wrap: break-word;
  font-weight: 500;
`;

const NewTopicSection = styled.div`
  margin-top: 3rem;
  padding: 2rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 15px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  text-align: center;
`;

const NewTopicTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1rem;
  font-family: 'Inter', sans-serif;
  color: #e2e8f0;
`;

const NewTopicDescription = styled.p`
  font-size: 1rem;
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 1.5rem;
  font-family: 'Inter', sans-serif;
`;

const NewTopicForm = styled.form`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  max-width: 500px;
  margin: 0 auto;
`;

const NewTopicInput = styled.input`
  width: 100%;
  padding: 1rem 1.5rem;
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 10px;
  background: rgba(0, 0, 0, 0.3);
  color: white;
  font-size: 1rem;
  font-family: 'Inter', sans-serif;
  transition: all 0.3s ease;
  
  &:focus {
    outline: none;
    border-color: #c084fc;
    box-shadow: 0 0 0 3px rgba(192, 132, 252, 0.1);
  }
  
  &::placeholder {
    color: rgba(255, 255, 255, 0.5);
  }
`;

const NewTopicButton = styled.button`
  padding: 1rem 2rem;
  background: linear-gradient(135deg, #ff6ac1, #38bdf8);
  border: none;
  border-radius: 10px;
  color: white;
  font-size: 1rem;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255, 106, 193, 0.3);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const StyledImage = styled.img`
  width: 350px;
  height: 350px;
  object-fit: cover;
  border-radius: 15px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  border: 2px solid rgba(255, 255, 255, 0.1);
  transition: all 0.3s ease;
  
  &:hover {
    transform: scale(1.05);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
  }
  
  @media (max-width: 768px) {
    width: 280px;
    height: 280px;
  }
  
  @media (max-width: 480px) {
    width: 250px;
    height: 250px;
  }
`;

const FileCard = styled.div`
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;

  &:hover {
    transform: translateY(-3px);
    border-color: rgba(192, 132, 252, 0.3);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
  }

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ff6ac1, #38bdf8);
  }
`;

const FileType = styled.div`
  position: absolute;
  top: 1rem;
  right: 1rem;
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 600;
  color: white;
  font-family: 'Inter', sans-serif;
`;

const ImageType = styled(FileType)`
  background: linear-gradient(135deg, #ff6ac1, #38bdf8);
`;

const JsonType = styled(FileType)`
  background: linear-gradient(135deg, #38bdf8, #c084fc);
`;

const MarkdownType = styled(FileType)`
  background: linear-gradient(135deg, #c084fc, #ff6ac1);
`;

const FileName = styled.h3`
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: white;
  font-family: 'Inter', sans-serif;
  padding-right: 4rem;
  word-break: break-word;
`;

const FileUrl = styled.div`
  font-size: 0.7rem;
  color: rgba(255, 255, 255, 0.5);
  margin-top: 0.5rem;
  word-break: break-all;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
`;

const FileUrlsSection = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: 2rem;
`;

const FileUrlsTitle = styled.h3`
  font-size: 1.2rem;
  font-weight: 600;
  color: #c084fc;
  margin-bottom: 1rem;
  font-family: 'Inter', sans-serif;
`;

const FileUrlsList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 0.5rem;
`;

const FileUrlItem = styled.li`
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.7);
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  word-break: break-all;
  padding: 0.3rem 0;
  border-left: 2px solid rgba(192, 132, 252, 0.3);
  padding-left: 0.5rem;
`;

const FilePreview = styled.div`
  background: rgba(0, 0, 0, 0.3);
  border-radius: 10px;
  padding: 1rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.8rem;
  line-height: 1.4;
  color: #e2e8f0;
  max-height: 200px;
  overflow-y: auto;
  white-space: pre-wrap;
`;

const ImagePreview = styled.div`
  width: 100%;
  height: 150px;
  background: linear-gradient(135deg, rgba(255, 106, 193, 0.1), rgba(56, 189, 248, 0.1));
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 2px dashed rgba(255, 255, 255, 0.2);
  color: rgba(255, 255, 255, 0.6);
  font-family: 'Inter', sans-serif;
  font-size: 0.9rem;
  overflow: hidden;
`;

const GeneratedImage = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 8px;
`;

const ImageError = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgba(239, 68, 68, 0.1);
  border: 2px dashed rgba(239, 68, 68, 0.3);
  color: #fca5a5;
  font-family: 'Inter', sans-serif;
  font-size: 0.8rem;
  text-align: center;
  padding: 0.5rem;
`;

const ErrorDetails = styled.div`
  background: rgba(0, 0, 0, 0.3);
  border-radius: 5px;
  padding: 0.5rem;
  margin-top: 0.5rem;
  font-size: 0.7rem;
  max-width: 100%;
  word-break: break-all;
`;

const SectionTitle = styled.h2`
  font-size: 2rem;
  font-weight: 700;
  margin: 2rem 0 1rem 0;
  color: #c084fc;
  font-family: 'Inter', sans-serif;
  text-align: center;
`;

const TopicDisplay = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin: 1rem 0;
  text-align: center;
`;

const TopicLabel = styled.div`
  font-size: 0.9rem;
  opacity: 0.7;
  margin-bottom: 0.5rem;
  font-family: 'Inter', sans-serif;
`;

const TopicValue = styled.div`
  font-size: 1.5rem;
  font-weight: 600;
  color: #ff6ac1;
  font-family: 'Inter', sans-serif;
`;

const FormContainer = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin-bottom: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
`;

const FormGroup = styled.div`
  margin-bottom: 1.5rem;
`;

const Label = styled.label`
  display: block;
  font-size: 1rem;
  font-weight: 600;
  color: #c084fc;
  margin-bottom: 0.5rem;
  font-family: 'Inter', sans-serif;
`;

const Input = styled.input`
  width: 100%;
  padding: 1rem;
  border-radius: 10px;
  border: 2px solid rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.1);
  color: white;
  font-size: 1rem;
  font-family: 'Inter', sans-serif;
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border-color: #c084fc;
    box-shadow: 0 0 0 3px rgba(192, 132, 252, 0.2);
  }

  &::placeholder {
    color: rgba(255, 255, 255, 0.5);
  }
`;

const SubmitButton = styled.button`
  background: linear-gradient(135deg, #ff6ac1, #38bdf8);
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 50px;
  font-weight: 600;
  font-size: 1.1rem;
  font-family: 'Inter', sans-serif;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(255, 106, 193, 0.3);
  width: 100%;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 106, 193, 0.4);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const ResponseContainer = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
`;

const ResponseTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 600;
  color: #c084fc;
  margin-bottom: 1rem;
  font-family: 'Inter', sans-serif;
`;

const ResponseContent = styled.div`
  background: rgba(0, 0, 0, 0.3);
  border-radius: 10px;
  padding: 1rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.9rem;
  line-height: 1.5;
  white-space: pre-wrap;
  overflow-x: auto;
  color: #e2e8f0;
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top: 2px solid #c084fc;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 10px;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ErrorMessage = styled.div`
  background: rgba(239, 68, 68, 0.2);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #fca5a5;
  padding: 1rem;
  border-radius: 10px;
  margin-top: 1rem;
  font-family: 'Inter', sans-serif;
`;

const SuccessMessage = styled.div`
  background: rgba(34, 197, 94, 0.2);
  border: 1px solid rgba(34, 197, 94, 0.3);
  color: #86efac;
  padding: 1rem;
  border-radius: 10px;
  margin-top: 1rem;
  font-family: 'Inter', sans-serif;
`;

const ResultsTextBox = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin: 2rem 0;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
`;

const ResultsTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 600;
  color: #c084fc;
  margin-bottom: 1.5rem;
  font-family: 'Inter', sans-serif;
  text-align: center;
`;

const ResultsContent = styled.div`
  background: rgba(0, 0, 0, 0.3);
  border-radius: 10px;
  padding: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 1rem;
  line-height: 1.6;
  white-space: pre-wrap;
  overflow-x: auto;
  color: #e2e8f0;
  max-height: 400px;
  overflow-y: auto;
`;

const BackButton = styled.button`
  background: linear-gradient(135deg, #ff6ac1, #38bdf8);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 25px;
  font-weight: 600;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  margin-bottom: 2rem;
  font-family: 'Inter', sans-serif;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(255, 106, 193, 0.3);
  }
`;

function ApiGenerator() {
  const location = useLocation();
  const { topicName, topicData } = location.state || {};
  
  const [topic, setTopic] = useState(topicName || '');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [imageStates, setImageStates] = useState({});
  const [fileContents, setFileContents] = useState({});
  const [imageCaptions, setImageCaptions] = useState([]);
  const [newTopic, setNewTopic] = useState('');
  const [newTopicLoading, setNewTopicLoading] = useState(false);

  // Update topic when location state changes and auto-generate if topic is provided
  useEffect(() => {
    if (topicName) {
      setTopic(topicName);
      // Auto-generate content when topic is provided
      handleAutoGenerate(topicName);
    }
  }, [topicName]);

  const handleAutoGenerate = async (topicName) => {
    if (!topicName.trim()) {
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');
    setResponse(null);
    setImageStates({});
    setFileContents({});

    try {
      const response = await fetch('https://meta-project-23px.onrender.com/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic: topicName.trim() }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('API Response:', data);
      console.log('Files in response:', data.files);
      setResponse(data);
      setSuccess('Images and content generated successfully!');
    } catch (err) {
      setError(`Error: ${err.message}`);
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!topic.trim()) {
      setError('Please enter a topic');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');
    setResponse(null);
    setImageStates({});
    setFileContents({});

    try {
      const response = await fetch('https://meta-project-23px.onrender.com/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic: topic.trim() }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('API Response:', data);
      console.log('Files in response:', data.files);
      setResponse(data);
      setSuccess('Images and content generated successfully!');
    } catch (err) {
      setError(`Error: ${err.message}`);
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleNewTopicSubmit = async (e) => {
    e.preventDefault();
    console.log('New topic form submitted with:', newTopic);
    
    if (!newTopic.trim()) {
      setError('Please enter a topic');
      return;
    }

    // Store the new topic before clearing
    const topicToGenerate = newTopic.trim();
    
    setNewTopicLoading(true);
    setError('');
    setSuccess('');
    setResponse(null);
    setImageStates({});
    setFileContents({});
    setImageCaptions([]);
    setNewTopic(''); // Clear the input immediately

    try {
      console.log('Making API call for topic:', topicToGenerate);
      setSuccess('Starting content generation... This may take 2-3 minutes.');
      
      const response = await fetch('https://meta-project-23px.onrender.com/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic: topicToGenerate }),
      });

      console.log('API response status:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('API Response:', data);
      console.log('Files in response:', data.files);
      
      // Update the topic state to show the new topic
      setTopic(topicToGenerate);
      setResponse(data);
      setSuccess('Content generated successfully!');
      
      // Extract captions from the report
      if (data.files) {
        const reportFile = data.files.find(file => file.includes('imagegen_report_') && file.includes('.md'));
        if (reportFile) {
          await fetchFileContent(reportFile);
        }
      }
    } catch (err) {
      console.error('API Error details:', err);
      setError(`Error: ${err.message}`);
      console.error('API Error:', err);
    } finally {
      setNewTopicLoading(false);
    }
  };

  const getFileType = (fileName) => {
    if (fileName.endsWith('.png') || fileName.endsWith('.jpg') || fileName.endsWith('.jpeg')) {
      return 'image';
    } else if (fileName.endsWith('.json')) {
      return 'json';
    } else if (fileName.endsWith('.md')) {
      return 'markdown';
    }
    return 'other';
  };

  const getFileUrl = (fileName) => {
    // Check if fileName already contains the full URL
    if (fileName.startsWith('http')) {
      return fileName;
    }
    return `https://meta-project-23px.onrender.com/files/${fileName}`;
  };

  const handleImageLoad = (fileName) => {
    setImageStates(prev => ({
      ...prev,
      [fileName]: { loaded: true, error: false }
    }));
  };

  const handleImageError = (fileName) => {
    console.error(`Image failed to load: ${fileName}`);
    setImageStates(prev => ({
      ...prev,
      [fileName]: { loaded: false, error: true, errorMessage: 'Image failed to load' }
    }));
  };

  const extractSummaryFromReport = (reportContent) => {
    if (!reportContent) return 'No summary available.';
    
    const lines = reportContent.split('\n');
    
    // Look for "**Summary:**" specifically
    for (const line of lines) {
      const trimmedLine = line.trim();
      
      // Look for "**Summary:**" pattern
      if (trimmedLine.includes('**Summary:**')) {
        const summaryPart = trimmedLine.split('**Summary:**')[1];
        if (summaryPart && summaryPart.trim().length > 5) {
          return summaryPart.trim();
        }
      }
      
      // Also look for "Summary:" pattern (without **)
      if (trimmedLine.includes('Summary:') && !trimmedLine.includes('**Summary:**')) {
        const summaryPart = trimmedLine.split('Summary:')[1];
        if (summaryPart && summaryPart.trim().length > 5) {
          return summaryPart.trim();
        }
      }
    }
    
    // If no "Summary:" found, look for other summary indicators
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      if (line.toLowerCase().includes('overview') ||
          line.toLowerCase().includes('introduction')) {
        
        // Get the next few lines as summary
        let summary = '';
        for (let j = i + 1; j < Math.min(i + 5, lines.length); j++) {
          const summaryLine = lines[j].trim();
          if (summaryLine && !summaryLine.startsWith('#') && !summaryLine.startsWith('##')) {
            summary += summaryLine + ' ';
          }
        }
        
        if (summary.trim()) {
          return summary.trim();
        }
      }
    }
    
    // If no summary section found, take first meaningful paragraph
    for (const line of lines) {
      const trimmedLine = line.trim();
      if (trimmedLine && 
          trimmedLine.length > 50 && 
          !trimmedLine.startsWith('#') && 
          !trimmedLine.startsWith('##') &&
          !trimmedLine.includes('Caption') &&
          !trimmedLine.includes('Image Prompt') &&
          !trimmedLine.includes('Summary:')) {
        return trimmedLine;
      }
    }
    
    return 'Summary not available.';
  };

  const extractCaptionsFromReport = (reportContent) => {
    if (!reportContent) return [];
    
    const captions = [];
    
    // Split content into lines and look for "Caption 1:", "Caption 2:", etc.
    const lines = reportContent.split('\n');
    
    // Look for numbered captions (Caption 1:, Caption 2:, etc.)
    for (let i = 1; i <= 5; i++) {
      const captionPatterns = [
        `Caption ${i}:`,
        `**Caption ${i}:**`,
        `Caption ${i}.`,
        `**Caption ${i}.**`
      ];
      
      for (const pattern of captionPatterns) {
        const matchingLine = lines.find(line => 
          line.trim().includes(pattern)
        );
        
        if (matchingLine) {
          const captionPart = matchingLine.split(pattern)[1];
          if (captionPart && captionPart.trim().length > 5) {
            // Clean up the caption content
            let cleanedCaption = captionPart.trim()
              .replace(/^\*\*/, '')  // Remove leading **
              .replace(/\*\*$/, '')  // Remove trailing **
              .replace(/^["']/, '')  // Remove leading quotes
              .replace(/["']$/, '')  // Remove trailing quotes
              .replace(/^[-*]\s*/, '')  // Remove leading bullet points
              .replace(/^\d+\.\s*/, '')  // Remove leading numbers
              .trim();
            
            if (cleanedCaption.length > 5) {
              captions[i - 1] = cleanedCaption; // Place in correct position (0-4)
              break; // Found this caption, move to next number
            }
          }
        }
      }
    }
    
    // If numbered captions not found, fallback to generic "Caption:" search
    if (captions.filter(c => c).length === 0) {
      lines.forEach(line => {
        const trimmedLine = line.trim();
        
        // Look for generic "Caption:" patterns
        if (trimmedLine.includes('Caption:') && !trimmedLine.match(/Caption \d+:/)) {
          const captionPart = trimmedLine.split('Caption:')[1];
          if (captionPart && captionPart.trim().length > 5) {
            let cleanedCaption = captionPart.trim()
              .replace(/^\*\*/, '')
              .replace(/\*\*$/, '')
              .replace(/^["']/, '')
              .replace(/["']$/, '')
              .replace(/^[-*]\s*/, '')
              .replace(/^\d+\.\s*/, '')
              .trim();
            
            if (cleanedCaption.length > 5) {
              captions.push(cleanedCaption);
            }
          }
        }
      });
    }
    
    // Filter out any remaining unwanted content
    const filteredCaptions = captions.filter(caption => 
      caption && 
      !caption.includes('Image Prompt:') && 
      !caption.includes('**Image Prompt:**') &&
      caption.length > 5
    );
    
    // Return up to 5 captions in order
    return filteredCaptions.slice(0, 5);
  };

  const fetchFileContent = async (fileName) => {
    try {
      console.log(`Fetching file: ${getFileUrl(fileName)}`);
      const response = await fetch(getFileUrl(fileName), {
        method: 'GET',
        headers: {
          'Accept': '*/*',
        },
      });
      
      console.log(`Response status for ${fileName}:`, response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const content = await response.text();
      console.log(`Content loaded for ${fileName}:`, content.substring(0, 100) + '...');
      
      setFileContents(prev => ({
        ...prev,
        [fileName]: { content, loaded: true, error: false }
      }));
      
      // Extract captions if this is the markdown report
      if (fileName.includes('imagegen_report_') && fileName.includes('.md')) {
        const captions = extractCaptionsFromReport(content);
        setImageCaptions(captions);
        console.log('Extracted captions:', captions);
      }
    } catch (error) {
      console.error(`Error fetching ${fileName}:`, error);
      setFileContents(prev => ({
        ...prev,
        [fileName]: { content: null, loaded: false, error: true, errorMessage: error.message }
      }));
    }
  };

  // Fetch file contents when response changes
  React.useEffect(() => {
    if (response && response.files) {
      console.log('Response received:', response);
      console.log('Files to fetch:', response.files);
      
      response.files.forEach(fileName => {
        const fileType = getFileType(fileName);
        if (fileType !== 'image' && !fileContents[fileName]) {
          fetchFileContent(fileName);
        }
      });
    }
  }, [response]);

  const renderImage = (fileName, index) => {
    const imageState = imageStates[fileName];
    const caption = imageCaptions[index] || `Generated image ${index + 1} for ${topic}`;
    
    return (
      <ImageContainer key={fileName}>
        {!imageState && (
          <div style={{ 
            width: '350px', 
            height: '350px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '15px',
            color: 'rgba(255, 255, 255, 0.6)'
          }}>
            Loading...
          </div>
        )}
        {imageState?.loaded && (
          <StyledImage 
            src={getFileUrl(fileName)}
            alt={fileName}
            onLoad={() => handleImageLoad(fileName)}
            onError={() => handleImageError(fileName)}
          />
        )}
        {imageState?.error && (
          <div style={{ 
            width: '350px', 
            height: '350px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            background: 'rgba(239, 68, 68, 0.1)',
            borderRadius: '15px',
            border: '2px dashed rgba(239, 68, 68, 0.3)',
            color: '#fca5a5',
            textAlign: 'center',
            padding: '1rem'
          }}>
            Failed to load image
          </div>
        )}
        {!imageState && (
          <StyledImage 
            src={getFileUrl(fileName)}
            alt={fileName}
            onLoad={() => handleImageLoad(fileName)}
            onError={() => handleImageError(fileName)}
            style={{ display: 'none' }}
          />
        )}
        
        {/* Image Caption */}
        {(imageState?.loaded || imageState?.error) && (
          <ImageCaption>
            {caption}
          </ImageCaption>
        )}
      </ImageContainer>
    );
  };

  const renderMarkdownContent = (fileName) => {
    const fileContent = fileContents[fileName];
    
    return (
      <FileUrlsSection>
        <FileUrlsTitle>Generated Report</FileUrlsTitle>
        <div style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          borderRadius: '10px',
          padding: '1rem',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          fontFamily: 'Monaco, Menlo, Ubuntu Mono, monospace',
          fontSize: '0.9rem',
          lineHeight: '1.5',
          whiteSpace: 'pre-wrap',
          color: '#e2e8f0',
          maxHeight: '300px',
          overflowY: 'auto'
        }}>
          {!fileContent && <div>Loading content...</div>}
          {fileContent?.loaded && fileContent.content && (
            <div>{fileContent.content}</div>
          )}
          {fileContent?.error && (
            <div style={{ color: '#fca5a5' }}>
              Failed to load content
              <ErrorDetails>
                URL: {getFileUrl(fileName)}
                <br />
                Error: {fileContent.errorMessage || 'Network error'}
              </ErrorDetails>
            </div>
          )}
        </div>
      </FileUrlsSection>
    );
  };

  return (
    <Container>
      <Content>
        <Header>
          <BackButton onClick={() => window.history.back()}>
            ← Back to Dashboard
          </BackButton>
          <Title>Image Generator</Title>
          <Subtitle>Generate images and content for any topic using our AI-powered API</Subtitle>
        </Header>

        {!topicName && (
          <FormContainer>
            <form onSubmit={handleSubmit}>
              <FormGroup>
                <Label htmlFor="topic">Enter Topic</Label>
                <Input
                  id="topic"
                  type="text"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="e.g., Purdue University, Technology Trends, etc."
                  disabled={loading}
                />
              </FormGroup>
              
              <SubmitButton type="submit" disabled={loading}>
                {loading ? (
                  <>
                    <LoadingSpinner />
                    Generating Content...
                  </>
                ) : (
                  'Generate Data'
                )}
              </SubmitButton>
            </form>
          </FormContainer>
        )}

        {topicName && loading && (
          <div style={{ 
            textAlign: 'center', 
            padding: '2rem',
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '15px',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            marginBottom: '2rem'
          }}>
            <LoadingSpinner />
            <div style={{ 
              fontSize: '1.2rem', 
              color: 'white', 
              marginTop: '1rem',
              fontFamily: 'Inter, sans-serif'
            }}>
              Generating AI content for "{topicName}"...
            </div>
            <div style={{ 
              fontSize: '0.9rem', 
              color: 'rgba(255, 255, 255, 0.7)', 
              marginTop: '0.5rem',
              fontFamily: 'Inter, sans-serif'
            }}>
              This may take 2-3 minutes. Please be patient while our AI creates your content! 🎨
            </div>
          </div>
        )}

        {error && <ErrorMessage>{error}</ErrorMessage>}
        {success && <SuccessMessage>{success}</SuccessMessage>}

        {response && (
          <>
            <TopicDisplay>
              <TopicLabel>Generated Topic</TopicLabel>
              <TopicValue>{response.topic}</TopicValue>
              
              {/* Summary Content */}
              {response.files && response.files
                .filter(fileName => fileName.includes('imagegen_report_') && fileName.includes('.md'))
                .map(fileName => {
                  const fileContent = fileContents[fileName];
                  return (
                    <div key={fileName} style={{
                      fontSize: '1rem',
                      lineHeight: '1.6',
                      color: '#e2e8f0',
                      fontFamily: 'Inter, sans-serif',
                      marginTop: '1rem',
                      textAlign: 'center'
                    }}>
                      {fileContent?.loaded && fileContent.content ? (
                        extractSummaryFromReport(fileContent.content)
                      ) : (
                        'Loading summary...'
                      )}
                    </div>
                  );
                })}
            </TopicDisplay>

            <ImageGrid>
              {(() => {
                const imageFiles = response.files ? response.files.filter(fileName => 
                  fileName.includes('img_') && fileName.includes('.png')
                ) : [];
                
                if (imageFiles.length === 5) {
                  // Custom layout: 3 on top, 2 in between on bottom
                  return (
                    <>
                      {/* Top row: images 1, 2, 3 */}
                      {imageFiles.slice(0, 3).map((fileName, index) => renderImage(fileName, index))}
                      
                      {/* Bottom row: images 4, 5 positioned between top images */}
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'center', 
                        gap: '2rem',
                        width: '100%',
                        marginTop: '2rem'
                      }}>
                        {imageFiles.slice(3, 5).map((fileName, index) => renderImage(fileName, index + 3))}
                      </div>
                    </>
                  );
                } else {
                  // Fallback for any number of images
                  return imageFiles.map((fileName, index) => renderImage(fileName, index));
                }
              })()}
            </ImageGrid>

            {/* New Topic Input Section */}
            <NewTopicSection>
              <NewTopicTitle>Generate Content for Another Topic</NewTopicTitle>
              <NewTopicDescription>
                Found something else interesting? Enter a new topic below to generate more AI content.
              </NewTopicDescription>
              
              {newTopicLoading && (
                <div style={{
                  background: 'rgba(192, 132, 252, 0.1)',
                  border: '1px solid rgba(192, 132, 252, 0.3)',
                  borderRadius: '10px',
                  padding: '1rem',
                  marginBottom: '1rem',
                  textAlign: 'center',
                  color: '#c084fc',
                  fontFamily: 'Inter, sans-serif'
                }}>
                  <LoadingSpinner />
                  Generating content for "{newTopic}"... This may take 2-3 minutes.
                </div>
              )}
              
              <NewTopicForm onSubmit={handleNewTopicSubmit}>
                <NewTopicInput
                  type="text"
                  value={newTopic}
                  onChange={(e) => setNewTopic(e.target.value)}
                  placeholder="Enter a new topic (e.g., Artificial Intelligence, Climate Change, etc.)"
                  disabled={newTopicLoading}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleNewTopicSubmit(e);
                    }
                  }}
                />
                
                <NewTopicButton type="submit" disabled={newTopicLoading}>
                  {newTopicLoading ? (
                    <>
                      <LoadingSpinner />
                      Generating Content...
                    </>
                  ) : (
                    'Generate New Content'
                  )}
                </NewTopicButton>
              </NewTopicForm>
            </NewTopicSection>

          </>
        )}
      </Content>
    </Container>
  );
}

export default ApiGenerator; 