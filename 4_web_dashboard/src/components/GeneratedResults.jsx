import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import styled from 'styled-components';

const Container = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 2rem;
  font-family: 'Inter', sans-serif;
`;

const Content = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  color: white;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 3rem;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: 700;
  color: #c084fc;
  margin-bottom: 1rem;
  font-family: 'Inter', sans-serif;
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 2rem;
`;

const BackButton = styled.button`
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 0.8rem 1.5rem;
  border-radius: 25px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  margin-bottom: 2rem;
  font-family: 'Inter', sans-serif;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
  }
`;

const LoadingSpinner = styled.div`
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top: 3px solid #c084fc;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  animation: spin 1s linear infinite;
  margin-right: 0.5rem;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const SuccessMessage = styled.div`
  background: rgba(34, 197, 94, 0.2);
  border: 1px solid rgba(34, 197, 94, 0.3);
  color: #86efac;
  padding: 1rem;
  border-radius: 10px;
  margin: 1rem 0;
  text-align: center;
  font-weight: 500;
`;

const ErrorMessage = styled.div`
  background: rgba(239, 68, 68, 0.2);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #fca5a5;
  padding: 1rem;
  border-radius: 10px;
  margin: 1rem 0;
  text-align: center;
  font-weight: 500;
`;

const RetryButton = styled.button`
  background: linear-gradient(135deg, #f59e0b, #d97706);
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  margin-top: 0.5rem;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
  font-family: 'Inter', sans-serif;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
  }
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
  justify-content: center;
  align-items: center;
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

const ReportSection = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: 2rem;
`;

const UrlsSection = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: 2rem;
`;

const SectionTitle = styled.h3`
  font-size: 1.2rem;
  font-weight: 600;
  color: #c084fc;
  margin-bottom: 1rem;
  font-family: 'Inter', sans-serif;
`;

function GeneratedResults() {
  const location = useLocation();
  const navigate = useNavigate();
  const [generatedData, setGeneratedData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [imageStates, setImageStates] = useState({});
  const [fileContents, setFileContents] = useState({});

  const { topicName } = location.state || {};

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
    } catch (error) {
      console.error(`Error fetching ${fileName}:`, error);
      setFileContents(prev => ({
        ...prev,
        [fileName]: { loaded: false, error: true, errorMessage: error.message }
      }));
    }
  };

  const generateImagesForTopic = async (topicName, retryCount = 0) => {
    setLoading(true);
    setError('');
    setSuccess('');
    setGeneratedData(null);

    try {
      console.log(`Generating images for topic: "${topicName}" (attempt ${retryCount + 1})`);
      const requestBody = { topic: topicName.trim() };
      console.log('Request body:', requestBody);

      const response = await fetch('https://meta-project-23px.onrender.com/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        let errorDetails = `HTTP error! status: ${response.status}`;
        try {
          const errorText = await response.text();
          console.log('Error response body:', errorText);
          if (errorText) {
            errorDetails += ` - ${errorText}`;
          }
        } catch (textError) {
          console.log('Could not read error response body');
        }
        throw new Error(errorDetails);
      }

      const data = await response.json();
      console.log('Generated images for topic:', topicName, data);
      
      if (!data || !data.files) {
        throw new Error('Invalid response format from server');
      }
      
      setGeneratedData(data);
      setSuccess(`✅ Generated ${data.files.filter(f => f.includes('img_') && f.includes('.png')).length} images and ${data.files.filter(f => f.includes('.md')).length} reports for "${topicName}"!`);
    } catch (err) {
      console.error('Image generation error:', err);
      
      let errorMessage = `Error generating images: ${err.message}`;
      
      if (err.message.includes('500')) {
        if (retryCount < 2) {
          console.log(`Retrying for 500 error (attempt ${retryCount + 1})`);
          setTimeout(() => {
            generateImagesForTopic(topicName, retryCount + 1);
          }, 2000 * (retryCount + 1));
          return;
        } else {
          errorMessage = 'Server error (500) - The API server is experiencing issues. Please try again in a few minutes.';
        }
      } else if (err.message.includes('404')) {
        errorMessage = 'API endpoint not found (404) - Please check if the service is available.';
      } else if (err.message.includes('network')) {
        errorMessage = 'Network error - Please check your internet connection and try again.';
      } else if (err.message.includes('timeout') || err.message.includes('AbortError')) {
        errorMessage = 'Request timeout - The server is taking too long to respond. Please try again in a few minutes.';
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const renderGeneratedContent = () => {
    if (!generatedData || !generatedData.files) return null;

    const imageFiles = generatedData.files.filter(fileName => 
      fileName.includes('img_') && fileName.includes('.png')
    );

    const markdownFiles = generatedData.files.filter(fileName => 
      fileName.includes('imagegen_report_') && fileName.includes('.md')
    );

    if (imageFiles.length === 0) return null;

    return (
      <>
        {/* Images Section */}
        <div style={{ marginBottom: '3rem' }}>
          <h3 style={{ 
            fontSize: '1.5rem', 
            fontWeight: '600', 
            color: '#c084fc',
            textAlign: 'center',
            marginBottom: '2rem',
            fontFamily: 'Inter, sans-serif'
          }}>
            Generated Images ({imageFiles.length} images)
          </h3>
          <ImageGrid>
            {(() => {
              if (imageFiles.length === 5) {
                return (
                  <>
                    {/* Top row: images 1, 2, 3 */}
                    {imageFiles.slice(0, 3).map((fileName, index) => (
                      <ImageContainer key={fileName}>
                        {!imageStates[fileName] && (
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
                        {imageStates[fileName]?.loaded && (
                          <StyledImage 
                            src={getFileUrl(fileName)}
                            alt={fileName}
                            onLoad={() => handleImageLoad(fileName)}
                            onError={() => handleImageError(fileName)}
                          />
                        )}
                        {imageStates[fileName]?.error && (
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
                            fontSize: '0.9rem',
                            textAlign: 'center'
                          }}>
                            Image failed to load
                          </div>
                        )}
                        {!imageStates[fileName] && (
                          <StyledImage 
                            src={getFileUrl(fileName)}
                            alt={fileName}
                            onLoad={() => handleImageLoad(fileName)}
                            onError={() => handleImageError(fileName)}
                            style={{ display: 'none' }}
                          />
                        )}
                      </ImageContainer>
                    ))}
                    
                    {/* Bottom row: images 4, 5 positioned between top images */}
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'center', 
                      gap: '2rem',
                      width: '100%',
                      marginTop: '2rem'
                    }}>
                      {imageFiles.slice(3, 5).map((fileName, index) => (
                        <ImageContainer key={fileName}>
                          {!imageStates[fileName] && (
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
                          {imageStates[fileName]?.loaded && (
                            <StyledImage 
                              src={getFileUrl(fileName)}
                              alt={fileName}
                              onLoad={() => handleImageLoad(fileName)}
                              onError={() => handleImageError(fileName)}
                            />
                          )}
                          {imageStates[fileName]?.error && (
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
                              fontSize: '0.9rem',
                              textAlign: 'center'
                            }}>
                              Image failed to load
                            </div>
                          )}
                          {!imageStates[fileName] && (
                            <StyledImage 
                              src={getFileUrl(fileName)}
                              alt={fileName}
                              onLoad={() => handleImageLoad(fileName)}
                              onError={() => handleImageError(fileName)}
                              style={{ display: 'none' }}
                            />
                          )}
                        </ImageContainer>
                      ))}
                    </div>
                  </>
                );
              } else {
                return imageFiles.map((fileName, index) => (
                  <ImageContainer key={fileName}>
                    {!imageStates[fileName] && (
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
                    {imageStates[fileName]?.loaded && (
                      <StyledImage 
                        src={getFileUrl(fileName)}
                        alt={fileName}
                        onLoad={() => handleImageLoad(fileName)}
                        onError={() => handleImageError(fileName)}
                      />
                    )}
                    {imageStates[fileName]?.error && (
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
                        fontSize: '0.9rem',
                        textAlign: 'center'
                      }}>
                        Image failed to load
                      </div>
                    )}
                    {!imageStates[fileName] && (
                      <StyledImage 
                        src={getFileUrl(fileName)}
                        alt={fileName}
                        onLoad={() => handleImageLoad(fileName)}
                        onError={() => handleImageError(fileName)}
                        style={{ display: 'none' }}
                      />
                    )}
                  </ImageContainer>
                ));
              }
            })()}
          </ImageGrid>
        </div>

        {/* Markdown Report Section */}
        {markdownFiles.length > 0 && (
          <ReportSection>
            <SectionTitle>Generated Report</SectionTitle>
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
              {!fileContents[markdownFiles[0]] && (
                <div style={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                  Loading report content...
                </div>
              )}
              {fileContents[markdownFiles[0]]?.loaded && fileContents[markdownFiles[0]].content && (
                <div>{fileContents[markdownFiles[0]].content}</div>
              )}
              {fileContents[markdownFiles[0]]?.error && (
                <div style={{ color: '#fca5a5' }}>
                  Failed to load report content. URL: https://meta-project-23px.onrender.com/files/{markdownFiles[0]}
                </div>
              )}
            </div>
          </ReportSection>
        )}

        {/* Generated File URLs Section */}
        <UrlsSection>
          <SectionTitle>Generated File URLs</SectionTitle>
          <div style={{ 
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '0.5rem'
          }}>
            {generatedData.files.map(fileName => (
              <div key={fileName} style={{
                fontSize: '0.8rem',
                color: 'rgba(255, 255, 255, 0.7)',
                fontFamily: 'Monaco, Menlo, Ubuntu Mono, monospace',
                wordBreak: 'break-all',
                padding: '0.3rem 0',
                borderLeft: '2px solid rgba(192, 132, 252, 0.3)',
                paddingLeft: '0.5rem'
              }}>
                {`https://meta-project-23px.onrender.com/files/${fileName}`}
              </div>
            ))}
          </div>
        </UrlsSection>
      </>
    );
  };

  useEffect(() => {
    if (topicName) {
      generateImagesForTopic(topicName);
    }
  }, [topicName]);

  // Fetch file contents when generatedData changes
  useEffect(() => {
    if (generatedData && generatedData.files) {
      // Fetch content for markdown files
      generatedData.files.forEach(fileName => {
        if (fileName.includes('.md') && !fileContents[fileName]) {
          fetchFileContent(fileName);
        }
      });
    }
  }, [generatedData]);

  if (!topicName) {
    return (
      <Container>
        <Content>
          <Header>
            <Title>Generated Results</Title>
            <Subtitle>No topic selected for generation</Subtitle>
            <BackButton onClick={() => navigate('/trend-strategy')}>
              ← Back to Dashboard
            </BackButton>
          </Header>
        </Content>
      </Container>
    );
  }

  return (
    <Container>
      <Content>
        <Header>
          <BackButton onClick={() => navigate('/trend-strategy')}>
            ← Back to Dashboard
          </BackButton>
          <Title>Generated Results</Title>
          <Subtitle>Generating content for: "{topicName}"</Subtitle>
        </Header>

        {loading && (
          <div style={{ 
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '50vh',
            textAlign: 'center',
            padding: '3rem',
            fontSize: '1.2rem',
            color: 'rgba(255, 255, 255, 0.8)'
          }}>
            <LoadingSpinner />
            <div style={{ marginTop: '1rem' }}>
              Generating images and reports...
            </div>
          </div>
        )}

        {success && <SuccessMessage>{success}</SuccessMessage>}
        
        {error && (
          <ErrorMessage>
            {error}
            <RetryButton onClick={() => generateImagesForTopic(topicName)}>
              Try Again
            </RetryButton>
          </ErrorMessage>
        )}

        {generatedData && renderGeneratedContent()}
      </Content>
    </Container>
  );
}

export default GeneratedResults; 