export interface InputData {
  gaokao_score: number;
  target_city: string;
  arts_science: string;
  family_situation?: string;
  risk_preference?: string;
}

export interface IndustryDistributionItem {
  industry: string;
  percentage: number;
}

export interface EmploymentData {
  employment_rate: number;
  avg_salary_bachelor: number;
  avg_salary_master: number;
  salary_growth_rate: number;
  top_employers: string[];
  industry_distribution: IndustryDistributionItem[];
}

export interface GraduatePublicExam {
  graduate_school_tendency: number;
  graduate_roi: string;
  graduate_school_recommendation: string;
  civil_service_tendency: number;
  civil_service_positions: number;
  civil_service_recommendation: string;
}

export interface AiResistanceIndex {
  score: number;
  level: string;
  color: string;
  interpretation: string;
  future_outlook: string;
  automation_probability: number;
}

export interface University {
  name: string;
  admission_probability: number;
  min_score_last_year: number;
  ranking: string;
}

export interface CareerPath {
  entry_level: string;
  mid_level: string;
  senior_level: string;
  avg_years_to_mid: number;
}

export interface MajorRecommendation {
  rank: number;
  major_name: string;
  major_code: string;
  category: string;
  match_score: number;
  match_reasons: string[];
  employment_data: EmploymentData;
  graduate_public_exam: GraduatePublicExam;
  ai_resistance_index: AiResistanceIndex;
  suggested_universities: University[];
  career_path: CareerPath;
  warnings: string[];
}

export interface RiskMajor {
  major: string;
  reason: string;
  ai_risk: number;
}

export interface OverratedMajor {
  major: string;
  social_popularity: string;
  reality_check: string;
  suggestion: string;
}

export interface PersonalizedAdvice {
  overall_strategy: string;
  graduate_school_advice: string;
  civil_service_advice: string;
  career_planning: string;
  skill_development: string[];
}

export interface AlternativeScenarios {
  if_score_higher_50: string;
  if_consider_abroad: string;
  if_family_limited: string;
}

export interface AnalysisSummary {
  total_matched_majors: number;
  avg_employment_rate: number;
  avg_ai_risk_index: number;
  recommendation_confidence: number;
}

export interface AnalysisResult {
  analysis_summary: AnalysisSummary;
  top_recommendations: MajorRecommendation[];
  risk_analysis: {
    high_risk_majors: RiskMajor[];
    overrated_majors: OverratedMajor[];
  };
  personalized_advice: PersonalizedAdvice;
  alternative_scenarios: AlternativeScenarios;
  data_sources: string[];
  disclaimer: string;
}