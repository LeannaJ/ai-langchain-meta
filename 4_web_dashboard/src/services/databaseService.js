import Papa from 'papaparse';

// Database service functions
export const databaseService = {
  // Get CSV file path based on content type
  getCsvFilePath: (contentType) => {
    switch (contentType) {
      case 'video':
        return '/consolidated_scores_w_crossbonus_video_2025-07-29_125911.csv';
      case 'image':
        return '/consolidated_scores_w_crossbonus_image_2025-07-29_125911.csv';
      case 'text':
        return '/consolidated_scores_w_crossbonus_text_2025-07-29_125911.csv';
      default:
        return '/consolidated_scores_w_crossbonus_2025-07-29_125911.csv';
    }
  },

  // Load categories from CSV
  loadCategories: async (contentType) => {
    try {
      const csvPath = databaseService.getCsvFilePath(contentType);
      console.log('Loading categories from CSV:', csvPath);
      
      const response = await fetch(csvPath);
      const csvText = await response.text();
      
      const result = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true
      });
      
      if (result.errors.length > 0) {
        console.error('CSV parsing errors:', result.errors);
        return [{ value: 'all', label: 'All Categories' }];
      }
      
      console.log('Raw categories data:', result.data);
      
      // Extract unique categories
      const categories = [...new Set(result.data
        .filter(d => d.category && d.category.trim() !== '')
        .map(d => d.category.trim())
      )].sort();

      // Add "All Categories" option
      const allCategories = [
        { value: 'all', label: 'All Categories' },
        ...categories.map(cat => ({ value: cat, label: cat }))
      ];

      return allCategories;
    } catch (error) {
      console.error('Error loading categories:', error);
      return [{ value: 'all', label: 'All Categories' }];
    }
  },

  // Load trending topics from CSV
  loadTrendingTopics: async (contentType, topicCategory = 'all') => {
    try {
      const csvPath = databaseService.getCsvFilePath(contentType);
      console.log('Loading trending topics from CSV:', csvPath);
      
      const response = await fetch(csvPath);
      const csvText = await response.text();
      
      const result = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true
      });
      
      if (result.errors.length > 0) {
        console.error('CSV parsing errors:', result.errors);
        return [];
      }
      
      let data = result.data;
      
      // Filter by category if not "all"
      if (topicCategory !== 'all') {
        data = data.filter(d => d.category === topicCategory);
      }

      // Remove duplicates based on group_name and keep the highest score
      const uniqueGroups = new Map();
      data.forEach(d => {
        const groupName = d.group_name?.trim();
        const score = Number(d.final_trend_score);

        if (groupName && !isNaN(score)) {
          if (!uniqueGroups.has(groupName) || score > uniqueGroups.get(groupName).score) {
            uniqueGroups.set(groupName, {
              group_name: groupName,
              final_trend_score: score,
              category: d.category
            });
          }
        }
      });

      // Process the data
      const processed = Array.from(uniqueGroups.values())
        .sort((a, b) => b.final_trend_score - a.final_trend_score)
        .slice(0, 10) // Show top 10 trending topics
        .map((d, index) => ({
          id: index + 1,
          name: d.group_name,
          trendScore: d.final_trend_score.toFixed(2),
          category: d.category
        }));

      return processed;
    } catch (error) {
      console.error('Error loading trending topics:', error);
      return [];
    }
  },

  // Load related keywords from CSV
  loadRelatedKeywords: async (groupName, contentType = 'all') => {
    try {
      const csvPath = databaseService.getCsvFilePath(contentType);
      console.log('Loading related keywords from CSV:', csvPath);
      
      const response = await fetch(csvPath);
      const csvText = await response.text();
      
      const result = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true
      });
      
      if (result.errors.length > 0) {
        console.error('CSV parsing errors:', result.errors);
        return [];
      }
      
      const data = result.data
        .filter(d => d.group_name === groupName && d.keyword && d.keyword.trim() !== '')
        .sort((a, b) => Number(b.final_trend_score) - Number(a.final_trend_score))
        .slice(0, 10);

      return data.map(d => ({
        keyword: d.keyword.trim(),
        score: Number(d.final_trend_score).toFixed(2)
      }));
    } catch (error) {
      console.error('Error loading related keywords:', error);
      return [];
    }
  },

  // Load data for main page bubble chart from CSV
  loadBubbleChartData: async () => {
    try {
      const csvPath = '/consolidated_scores_w_crossbonus_2025-07-29_125911.csv';
      console.log('Loading bubble chart data from CSV:', csvPath);
      
      const response = await fetch(csvPath);
      const csvText = await response.text();
      
      const result = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true
      });
      
      if (result.errors.length > 0) {
        console.error('CSV parsing errors:', result.errors);
        return [];
      }
      
      const data = result.data.filter(d => d.group_name && d.final_trend_score);
      console.log('Raw data from CSV:', data);

      // Remove duplicates based on group_name and keep the highest score
      const uniqueGroups = new Map();
      data.forEach(d => {
        const groupName = d.group_name?.trim();
        const score = Number(d.final_trend_score);

        if (groupName && !isNaN(score)) {
          if (!uniqueGroups.has(groupName) || score > uniqueGroups.get(groupName).score) {
            uniqueGroups.set(groupName, {
              group_name: groupName,
              final_trend_score: score
            });
          }
        }
      });

      const sorted = Array.from(uniqueGroups.values())
        .sort((a, b) => b.final_trend_score - a.final_trend_score)
        .slice(0, 15) // Display top 15 bubbles
        .map((d, i) => ({
          keyword: d.group_name,
          score: d.final_trend_score,
          rank: i + 1
        }));

      console.log('Processed bubble chart data:', sorted);
      return sorted;
    } catch (error) {
      console.error('Error loading bubble chart data:', error);
      return [];
    }
  }
};

export default databaseService; 