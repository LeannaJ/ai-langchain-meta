import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import databaseService from '../services/databaseService';

const Container = styled.div`
  min-height: 100vh;
  padding: 80px 2rem 2rem;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: white;
`;

const Content = styled.div`
  max-width: 1400px;
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

const FilterSection = styled.div`
  display: flex;
  justify-content: center;
  gap: 2rem;
  margin-bottom: 3rem;
  flex-wrap: wrap;
`;

const FilterGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const FilterLabel = styled.label`
  font-size: 0.9rem;
  font-weight: 600;
  color: #c084fc;
  font-family: 'Inter', sans-serif;
`;

const FilterSelect = styled.select`
  padding: 0.75rem 1rem;
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 10px;
  background: rgba(0, 0, 0, 0.3);
  color: white;
  font-size: 1rem;
  font-family: 'Inter', sans-serif;
  cursor: pointer;
  transition: all 0.3s ease;
  min-width: 150px;
  
  &:focus {
    outline: none;
    border-color: #c084fc;
    box-shadow: 0 0 0 3px rgba(192, 132, 252, 0.1);
  }
  
  option {
    background: #1a1a2e;
    color: white;
  }
`;

const TrendingTopicsSection = styled.div`
  margin-bottom: 3rem;
`;

const SectionTitle = styled.h2`
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 2rem;
  text-align: center;
  color: #e2e8f0;
  font-family: 'Inter', sans-serif;
`;

const TrendingTopicsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1.5rem;
  margin-bottom: 3rem;
  opacity: ${props => props.$loading ? 0.6 : 1};
  transition: opacity 0.3s ease;
`;

const TopicCard = styled.div`
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  border-radius: 15px;
  padding: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    border-color: rgba(255, 255, 255, 0.2);
  }
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ff6ac1, #38bdf8);
    transform: scaleX(0);
    transition: transform 0.3s ease;
  }
  
  &:hover::before {
    transform: scaleX(1);
  }
`;

const TopicRank = styled.div`
  position: absolute;
  top: 1rem;
  right: 1rem;
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #ff6ac1, #38bdf8);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 1.2rem;
  color: white;
  font-family: 'Inter', sans-serif;
`;

const TopicName = styled.h3`
  font-size: 1.3rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #e2e8f0;
  font-family: 'Inter', sans-serif;
  line-height: 1.4;
  word-wrap: break-word;
  overflow-wrap: break-word;
  hyphens: auto;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
`;

const TopicScore = styled.div`
  font-size: 1.1rem;
  color: #c084fc;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
`;

const HoverOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.3s ease;
  border-radius: 15px;
  
  ${TopicCard}:hover & {
    opacity: 1;
  }
`;

const HoverText = styled.div`
  color: white;
  font-size: 1.1rem;
  font-weight: 600;
  text-align: center;
  font-family: 'Inter', sans-serif;
`;

const InfoIcon = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  background: rgba(192, 132, 252, 0.2);
  border: 2px solid #c084fc;
  border-radius: 50%;
  color: #c084fc;
  font-size: 14px;
  font-weight: 700;
  margin-left: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  
  &:hover {
    background: rgba(192, 132, 252, 0.3);
    transform: scale(1.1);
  }
`;

const Tooltip = styled.div`
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(255, 255, 255, 0.85);
  color: #1a1a2e;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 0.85rem;
  font-family: 'Inter', sans-serif;
  line-height: 1.4;
  text-align: center;
  z-index: 1000;
  opacity: 0;
  visibility: hidden;
  transition: all 0.3s ease;
  margin-bottom: 8px;
  border: 1px solid rgba(192, 132, 252, 0.3);
  max-width: 600px;
  min-width: 450px;
  word-wrap: break-word;
  overflow-wrap: break-word;
  hyphens: auto;
  
  &::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: rgba(255, 255, 255, 0.85);
  }
`;

const LoadingSpinner = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 3rem;
  
  &::after {
    content: '';
    width: 40px;
    height: 40px;
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-top: 3px solid #c084fc;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const BarChartContainer = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 15px;
  padding: 2rem 2rem 2rem 20rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: 2rem;
`;

const BarChartTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 2rem;
  text-align: center;
  color: #e2e8f0;
  font-family: 'Inter', sans-serif;
`;

const BarChart = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  padding: 1rem 0;
  height: 500px;
`;

const Bar = styled.div`
  height: 40px;
  background: linear-gradient(to right, #ff6ac1, #38bdf8);
  border-radius: 0 8px 8px 0;
  min-height: 30px;
  position: relative;
  transition: all 0.3s ease;
  
  &:hover {
    transform: scaleX(1.02);
    box-shadow: 0 4px 20px rgba(255, 106, 193, 0.4);
  }
`;

const BarLabel = styled.div`
  position: absolute;
  left: -240px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 1.2rem;
  color: #e2e8f0;
  font-weight: 600;
  text-align: right;
  white-space: nowrap;
  font-family: 'Inter', sans-serif;
  max-width: 250px;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const BarValue = styled.div`
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 1.1rem;
  color: white;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
`;

const DetailView = styled.div`
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  border-radius: 15px;
  padding: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin-bottom: 2rem;
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

const KeywordsList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1rem;
`;

const KeywordTag = styled.div`
  background: rgba(192, 132, 252, 0.2);
  border: 1px solid rgba(192, 132, 252, 0.3);
  border-radius: 20px;
  padding: 0.5rem 1rem;
  font-size: 0.9rem;
  color: #e2e8f0;
  font-family: 'Inter', sans-serif;
`;

function TrendStrategy() {
  const { topicId } = useParams();
  const navigate = useNavigate();
  
  const [topicCategory, setTopicCategory] = useState('all');
  const [contentType, setContentType] = useState('all');
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [isDetailView, setIsDetailView] = useState(false);
  const [trendingTopics, setTrendingTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [availableCategories, setAvailableCategories] = useState([]);
  const [relatedKeywords, setRelatedKeywords] = useState([]);

  // Load categories when content type changes
  const loadCategories = async (contentType) => {
    try {
      const categories = await databaseService.loadCategories(contentType);
      setAvailableCategories(categories);
    } catch (error) {
      console.error('Error loading categories:', error);
      setAvailableCategories([{ value: 'all', label: 'All Categories' }]);
    }
  };

  // Load trending topics
  const loadTrendingTopics = async (contentType) => {
    setLoading(true);
    // Add a small delay for smooth transitions
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      const processed = await databaseService.loadTrendingTopics(contentType, topicCategory);
      setTrendingTopics(processed);
    } catch (error) {
      console.error('Error loading trending topics:', error);
      setTrendingTopics([]);
    } finally {
      setLoading(false);
    }
  };

  // Load related keywords for detail view
  const loadRelatedKeywords = async (groupName, contentType) => {
    try {
      const sortedRelated = await databaseService.loadRelatedKeywords(groupName, contentType);
      setRelatedKeywords(sortedRelated);
    } catch (error) {
      console.error('Error loading related keywords:', error);
      setRelatedKeywords([]);
    }
  };

  // Handle topic selection for detail view
  const handleTopicClick = (topic) => {
    setSelectedTopic(topic);
    setIsDetailView(true);
    loadRelatedKeywords(topic.name, contentType);
  };

  // Handle filter changes
  const handleContentTypeChange = (newContentType) => {
    setContentType(newContentType);
    setTopicCategory('all'); // Reset category when content type changes
  };

  const handleCategoryChange = (newCategory) => {
    setTopicCategory(newCategory);
  };

  // Load data on component mount and filter changes
  useEffect(() => {
    loadCategories(contentType);
  }, [contentType]);

  useEffect(() => {
    loadTrendingTopics(contentType);
  }, [contentType, topicCategory]);

  // Handle URL parameters for detail view
  useEffect(() => {
    if (topicId) {
      const topic = trendingTopics.find(t => t.id === parseInt(topicId));
      if (topic) {
        handleTopicClick(topic);
      }
    }
  }, [topicId, trendingTopics]);

  // Calculate bar chart data
  const barChartData = trendingTopics.map((topic, index) => ({
    name: topic.name,
    score: parseFloat(topic.trendScore),
    rank: index + 1
  }));

  const maxScore = Math.max(...barChartData.map(d => d.score), 1);
  // For scores below 0.35, use a max of 0.4 to make bars much longer
  const adjustedMaxScore = 0.4;

  if (isDetailView && selectedTopic) {
    return (
      <Container>
        <Content>
          <BackButton onClick={() => {
            setIsDetailView(false);
            setSelectedTopic(null);
            navigate('/trend-strategy');
          }}>
            ← Back to Trends
          </BackButton>
          
          <DetailView>
            <Title>{selectedTopic.name}</Title>
            <Subtitle>Trend Score: {selectedTopic.trendScore}</Subtitle>
            
            <SectionTitle>Related Keywords</SectionTitle>
            <KeywordsList>
              {relatedKeywords.map((keyword, index) => (
                <KeywordTag key={index}>
                  {keyword.keyword} ({keyword.score})
                </KeywordTag>
              ))}
            </KeywordsList>
          </DetailView>
        </Content>
      </Container>
    );
  }

  return (
    <Container>
      <Content>
        <Header>
          <Title>Trend Strategy Dashboard</Title>
          <Subtitle>Discover trending topics and generate content</Subtitle>
        </Header>

        <FilterSection>
          <FilterGroup>
            <FilterLabel>Content Type</FilterLabel>
            <FilterSelect 
              value={contentType} 
              onChange={(e) => handleContentTypeChange(e.target.value)}
            >
              <option value="all">All Content</option>
              <option value="video">Video</option>
              <option value="image">Image</option>
              <option value="text">Text</option>
            </FilterSelect>
          </FilterGroup>

          <FilterGroup>
            <FilterLabel>Topic Category</FilterLabel>
            <FilterSelect 
              value={topicCategory} 
              onChange={(e) => handleCategoryChange(e.target.value)}
            >
              {availableCategories.map((category, index) => (
                <option key={index} value={category.value}>
                  {category.label}
                </option>
              ))}
            </FilterSelect>
          </FilterGroup>
        </FilterSection>

        <TrendingTopicsSection>
          <SectionTitle>Top 10 Trending Topics</SectionTitle>
          
          {loading ? (
            <LoadingSpinner />
          ) : (
            <>
              <TrendingTopicsGrid $loading={loading}>
                {trendingTopics.map((topic, index) => (
                  <TopicCard
                    key={`${contentType}-${topic.id}`}
                    onClick={() => navigate('/api-generator', {
                      state: {
                        topicName: topic.name,
                        topicData: topic
                      }
                    })}
                    style={{ animationDelay: `${index * 0.05}s` }}
                  >
                    <TopicRank>{index + 1}</TopicRank>
                    <TopicName>{topic.name}</TopicName>
                    <HoverOverlay>
                      <HoverText>Click to generate AI content</HoverText>
                    </HoverOverlay>
                  </TopicCard>
                ))}
              </TrendingTopicsGrid>

                                      {trendingTopics.length > 0 && (
                          <BarChartContainer>
                            <BarChartTitle>
                              Trend Score Comparison
                              <InfoIcon 
                                onMouseEnter={(e) => {
                                  e.currentTarget.querySelector('.tooltip').style.opacity = '1';
                                  e.currentTarget.querySelector('.tooltip').style.visibility = 'visible';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.querySelector('.tooltip').style.opacity = '0';
                                  e.currentTarget.querySelector('.tooltip').style.visibility = 'hidden';
                                }}
                              >
                                ?
                                <Tooltip className="tooltip">
                                  <div>Topics are ranked based on how widely</div>
                                  <div>discussed they are, how much engagement</div>
                                  <div>they get, and whether they appear across multiple social platforms.</div>
                                </Tooltip>
                              </InfoIcon>
                            </BarChartTitle>
                  <BarChart>
                    {barChartData.map((data, index) => (
                                                                   <Bar
                        key={index}
                        style={{ 
                          width: `${(data.score / adjustedMaxScore) * 100}%`,
                          minWidth: '20px'
                        }}
                      >
                        <BarValue>{data.score}</BarValue>
                        <BarLabel>{data.name}</BarLabel>
                      </Bar>
                    ))}
                  </BarChart>
                </BarChartContainer>
              )}
            </>
          )}
        </TrendingTopicsSection>
      </Content>
    </Container>
  );
}

export default TrendStrategy;
