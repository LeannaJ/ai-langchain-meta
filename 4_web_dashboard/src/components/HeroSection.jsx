import styled from 'styled-components';
import { Link } from 'react-router-dom';

const HeroContainer = styled.section`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-end;
  min-height: 35vh;
  padding-top: 2vh;
`;

const Title = styled.h1`
  font-size: 6rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  color: #fff;
  margin-bottom: 0;
  font-family: 'Inter', sans-serif;
`;

const TrendingText = styled.h2`
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #ff6ac1, #38bdf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 1.5rem;
  font-family: 'Inter', sans-serif;
  text-align: center;
  text-shadow: 0 0 30px rgba(255, 106, 193, 0.3);
  letter-spacing: 0.05em;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #ff6ac1, #38bdf8);
    border-radius: 2px;
  }
`;

const Description = styled.p`
  font-size: 1rem;
  color: #b0b8c1;
  margin-bottom: 2rem;
  font-family: 'Inter', sans-serif;
`;

const CTAButton = styled(Link)`
  background: linear-gradient(135deg, #ff6ac1, #38bdf8);
  color: white;
  padding: 1rem 2rem;
  border-radius: 50px;
  text-decoration: none;
  font-weight: 600;
  font-size: 1.1rem;
  font-family: 'Inter', sans-serif;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(255, 106, 193, 0.3);
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 106, 193, 0.4);
    color: white;
  }
`;

function HeroSection() {
  return (
    <HeroContainer>
      <TrendingText>What's trending today?</TrendingText>
      <Title>ASTRA</Title>
      <Description>Dig into global trends for your contents creation</Description>
      <CTAButton to="/trend-strategy">
        Let's dig into trend and content strategy!
      </CTAButton>
    </HeroContainer>
  );
}

export default HeroSection; 