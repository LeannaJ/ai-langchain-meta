import React from 'react';
import styled from 'styled-components';

const Container = styled.div`
  min-height: 100vh;
  padding: 80px 2rem 2rem;
  background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0f4c75 100%);
  color: white;
  position: relative;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
    pointer-events: none;
  }
`;

const Content = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 3rem;
`;

const Title = styled.h1`
  font-size: 3.5rem;
  font-weight: 900;
  margin-bottom: 1rem;
  font-family: 'Inter', sans-serif;
  background: linear-gradient(135deg, #ff6ac1, #38bdf8, #c084fc);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 0 30px rgba(255, 106, 193, 0.3);
  letter-spacing: -0.02em;
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  opacity: 0.9;
  font-family: 'Inter', sans-serif;
`;

const Section = styled.section`
  margin-bottom: 3rem;
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  padding: 2.5rem;
  border: 1px solid rgba(255, 255, 255, 0.15);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    border-color: rgba(255, 255, 255, 0.2);
  }
`;

const SectionTitle = styled.h2`
  font-size: 2.2rem;
  font-weight: 800;
  margin-bottom: 1.5rem;
  background: linear-gradient(135deg, #c084fc, #38bdf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-family: 'Inter', sans-serif;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 0;
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #c084fc, #38bdf8);
    border-radius: 2px;
  }
`;

const StepTitle = styled.h3`
  font-size: 1.4rem;
  font-weight: 700;
  margin: 1.5rem 0 1rem 0;
  color: #38bdf8;
  font-family: 'Inter', sans-serif;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &::before {
    content: "→";
    color: #ff6ac1;
    font-weight: 800;
  }
`;

const Text = styled.p`
  font-size: 1rem;
  line-height: 1.6;
  margin-bottom: 1rem;
  font-family: 'Inter', sans-serif;
  color: #e2e8f0;
`;

const Highlight = styled.span`
  background: linear-gradient(135deg, #ff6ac1, #38bdf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-weight: 600;
`;

const FeatureList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 1rem 0;
`;

const FeatureItem = styled.li`
  padding: 1rem;
  border-left: 3px solid #c084fc;
  padding-left: 1.5rem;
  margin-bottom: 0.75rem;
  font-family: 'Inter', sans-serif;
  color: #e2e8f0;
  background: rgba(192, 132, 252, 0.05);
  border-radius: 0 10px 10px 0;
  transition: all 0.3s ease;
  
  &:before {
    content: "✨";
    margin-right: 0.5rem;
    font-size: 1.1rem;
  }
  
  &:hover {
    background: rgba(192, 132, 252, 0.1);
    transform: translateX(5px);
  }
`;

const TipBox = styled.div`
  background: linear-gradient(135deg, rgba(192, 132, 252, 0.1), rgba(56, 189, 248, 0.1));
  border: 1px solid rgba(192, 132, 252, 0.3);
  border-radius: 15px;
  padding: 1.5rem;
  margin: 1.5rem 0;
  font-family: 'Inter', sans-serif;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #c084fc, #38bdf8);
  }
`;

const TipTitle = styled.div`
  font-weight: 600;
  color: #c084fc;
  margin-bottom: 0.5rem;
`;

const TipContent = styled.div`
  color: #e2e8f0;
  font-size: 0.95rem;
`;

const NavigationTip = styled.div`
  background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(255, 106, 193, 0.1));
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 15px;
  padding: 1.5rem;
  margin: 1.5rem 0;
  font-family: 'Inter', sans-serif;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #ff6ac1);
  }
`;

const NavigationTitle = styled.div`
  font-weight: 600;
  color: #38bdf8;
  margin-bottom: 0.5rem;
`;

const NavigationContent = styled.div`
  color: #e2e8f0;
  font-size: 0.95rem;
`;

function RawData() {
  return (
    <Container>
      <Content>
        <Header>
          <Title>ASTRA User Guide</Title>
          <Subtitle>Quick guide to using the ASTRA dashboard</Subtitle>
        </Header>

        <Section>
          <SectionTitle>🎯 What is ASTRA?</SectionTitle>
          <Text>
            ASTRA is a dashboard that shows trending topics and generates AI content. 
            You can discover what's popular and create images/content for any topic.
          </Text>
        </Section>

        <Section>
          <SectionTitle>🏠 Home Page</SectionTitle>
          
          <StepTitle>1. Explore the Bubble Chart</StepTitle>
          <Text>
            The main page shows trending topics as bubbles. <Highlight>Hover over bubbles</Highlight> to see them enlarge.
            Larger bubbles = higher trend scores.
          </Text>

          <StepTitle>2. Go to Dashboard</StepTitle>
          <Text>
            Click the <Highlight>"Let's dig into trend and content strategy!"</Highlight> button to see detailed trends.
          </Text>
        </Section>

        <Section>
          <SectionTitle>📊 Trend Dashboard</SectionTitle>
          
          <StepTitle>1. Use Filters</StepTitle>
          <FeatureList>
            <FeatureItem><Highlight>Content Type:</Highlight> Choose Video, Image, Text, or All</FeatureItem>
            <FeatureItem><Highlight>Topic Category:</Highlight> Filter by specific categories</FeatureItem>
          </FeatureList>

          <StepTitle>2. View Trending Topics</StepTitle>
          <Text>
            See top 10 trending topics as cards. <Highlight>Click any card</Highlight> to generate AI content for that topic.
          </Text>

          <StepTitle>3. Check the Bar Chart</StepTitle>
          <Text>
            The bar chart below shows trend scores visually - taller bars = higher scores.
          </Text>
        </Section>

        <Section>
          <SectionTitle>🎨 Generate AI Content</SectionTitle>
          
          <StepTitle>1. Start Generation</StepTitle>
          <FeatureList>
            <FeatureItem><Highlight>From Trend Cards:</Highlight> Click any trending topic card</FeatureItem>
            <FeatureItem><Highlight>Manual Input:</Highlight> Use the input box at the bottom of the generation page</FeatureItem>
          </FeatureList>

          <StepTitle>2. Wait for Results</StepTitle>
          <Text>
            Generation takes 2-3 minutes. You'll see a loading spinner with progress updates.
          </Text>

          <StepTitle>3. View Generated Content</StepTitle>
          <FeatureList>
            <FeatureItem>5 AI-generated images with captions</FeatureItem>
            <FeatureItem>Topic summary</FeatureItem>
            <FeatureItem>Input box to generate more content</FeatureItem>
          </FeatureList>

          <TipBox>
            <TipTitle>💡 Tip</TipTitle>
            <TipContent>
              Be specific with topics. Instead of "technology", try "artificial intelligence in healthcare".
            </TipContent>
          </TipBox>
        </Section>

        <Section>
          <SectionTitle>🧭 Navigation</SectionTitle>
          
          <NavigationTip>
            <NavigationTitle>🏠 Home</NavigationTitle>
            <NavigationContent>
              Click the "ASTRA" logo to return to the home page from anywhere.
            </NavigationContent>
          </NavigationTip>

          <NavigationTip>
            <NavigationTitle>📖 User Manual</NavigationTitle>
            <NavigationContent>
              You're here! Access via "User Manual" in the navigation.
            </NavigationContent>
          </NavigationTip>

          <NavigationTip>
            <NavigationTitle>🎨 Image Generator</NavigationTitle>
            <NavigationContent>
              Direct access to content generation. Also reached by clicking trend cards.
            </NavigationContent>
          </NavigationTip>
        </Section>


      </Content>
    </Container>
  );
}

export default RawData; 