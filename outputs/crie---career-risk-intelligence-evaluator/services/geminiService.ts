import { GoogleGenAI, Type, Schema } from "@google/genai";
import { InputData, AnalysisResult } from "../types";

const apiKey = process.env.API_KEY || "";
const ai = new GoogleGenAI({ apiKey });

// Define the schema for the Gemini response to ensure strict JSON structure
const responseSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    analysis_summary: {
      type: Type.OBJECT,
      properties: {
        total_matched_majors: { type: Type.INTEGER },
        avg_employment_rate: { type: Type.NUMBER },
        avg_ai_risk_index: { type: Type.NUMBER },
        recommendation_confidence: { type: Type.NUMBER },
      },
      required: ["total_matched_majors", "avg_employment_rate", "avg_ai_risk_index", "recommendation_confidence"]
    },
    top_recommendations: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          rank: { type: Type.INTEGER },
          major_name: { type: Type.STRING },
          major_code: { type: Type.STRING },
          category: { type: Type.STRING },
          match_score: { type: Type.NUMBER },
          match_reasons: { type: Type.ARRAY, items: { type: Type.STRING } },
          employment_data: {
            type: Type.OBJECT,
            properties: {
              employment_rate: { type: Type.NUMBER },
              avg_salary_bachelor: { type: Type.NUMBER },
              avg_salary_master: { type: Type.NUMBER },
              salary_growth_rate: { type: Type.NUMBER },
              top_employers: { type: Type.ARRAY, items: { type: Type.STRING } },
              industry_distribution: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    industry: { type: Type.STRING },
                    percentage: { type: Type.NUMBER }
                  },
                  required: ["industry", "percentage"]
                }
              },
            },
            required: ["employment_rate", "avg_salary_bachelor", "avg_salary_master", "salary_growth_rate", "top_employers", "industry_distribution"]
          },
          graduate_public_exam: {
            type: Type.OBJECT,
            properties: {
              graduate_school_tendency: { type: Type.NUMBER },
              graduate_roi: { type: Type.STRING },
              graduate_school_recommendation: { type: Type.STRING },
              civil_service_tendency: { type: Type.NUMBER },
              civil_service_positions: { type: Type.INTEGER },
              civil_service_recommendation: { type: Type.STRING },
            },
            required: ["graduate_school_tendency", "graduate_roi", "graduate_school_recommendation", "civil_service_tendency", "civil_service_positions", "civil_service_recommendation"]
          },
          ai_resistance_index: {
            type: Type.OBJECT,
            properties: {
              score: { type: Type.NUMBER },
              level: { type: Type.STRING },
              color: { type: Type.STRING },
              interpretation: { type: Type.STRING },
              future_outlook: { type: Type.STRING },
              automation_probability: { type: Type.NUMBER },
            },
            required: ["score", "level", "color", "interpretation", "future_outlook", "automation_probability"]
          },
          suggested_universities: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                name: { type: Type.STRING },
                admission_probability: { type: Type.NUMBER },
                min_score_last_year: { type: Type.INTEGER },
                ranking: { type: Type.STRING },
              },
              required: ["name", "admission_probability", "min_score_last_year", "ranking"]
            }
          },
          career_path: {
            type: Type.OBJECT,
            properties: {
              entry_level: { type: Type.STRING },
              mid_level: { type: Type.STRING },
              senior_level: { type: Type.STRING },
              avg_years_to_mid: { type: Type.NUMBER },
            },
            required: ["entry_level", "mid_level", "senior_level", "avg_years_to_mid"]
          },
          warnings: { type: Type.ARRAY, items: { type: Type.STRING } },
        },
        required: ["rank", "major_name", "match_score", "match_reasons", "employment_data", "graduate_public_exam", "ai_resistance_index", "suggested_universities", "career_path", "warnings"]
      }
    },
    risk_analysis: {
      type: Type.OBJECT,
      properties: {
        high_risk_majors: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              major: { type: Type.STRING },
              reason: { type: Type.STRING },
              ai_risk: { type: Type.NUMBER },
            },
            required: ["major", "reason", "ai_risk"]
          }
        },
        overrated_majors: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              major: { type: Type.STRING },
              social_popularity: { type: Type.STRING },
              reality_check: { type: Type.STRING },
              suggestion: { type: Type.STRING },
            },
            required: ["major", "social_popularity", "reality_check", "suggestion"]
          }
        }
      },
      required: ["high_risk_majors", "overrated_majors"]
    },
    personalized_advice: {
      type: Type.OBJECT,
      properties: {
        overall_strategy: { type: Type.STRING },
        graduate_school_advice: { type: Type.STRING },
        civil_service_advice: { type: Type.STRING },
        career_planning: { type: Type.STRING },
        skill_development: { type: Type.ARRAY, items: { type: Type.STRING } },
      },
      required: ["overall_strategy", "graduate_school_advice", "civil_service_advice", "career_planning", "skill_development"]
    },
    alternative_scenarios: {
      type: Type.OBJECT,
      properties: {
        if_score_higher_50: { type: Type.STRING },
        if_consider_abroad: { type: Type.STRING },
        if_family_limited: { type: Type.STRING },
      },
      required: ["if_score_higher_50", "if_consider_abroad", "if_family_limited"]
    },
    data_sources: { type: Type.ARRAY, items: { type: Type.STRING } },
    disclaimer: { type: Type.STRING },
  },
  required: ["analysis_summary", "top_recommendations", "risk_analysis", "personalized_advice", "alternative_scenarios", "data_sources", "disclaimer"]
};

export const generateCareerAnalysis = async (input: InputData): Promise<AnalysisResult> => {
  if (!apiKey) {
    throw new Error("API Key is missing. Please set REACT_APP_GEMINI_API_KEY.");
  }

  const prompt = `
    作为中国高考志愿填报与职业规划专家，请根据以下学生信息进行深度分析：
    
    学生分数: ${input.gaokao_score}
    意向城市: ${input.target_city}
    文理分科: ${input.arts_science}
    家庭情况: ${input.family_situation || "未提供"}
    风险偏好: ${input.risk_preference || "平衡型"}

    请基于2024年及未来的中国就业市场趋势、AI替代风险研究、历年各省录取数据，生成一份详细的专业评估报告。
    
    要求：
    1. 推荐Top 3-5个最适合的本科专业。
    2. 对每个专业进行AI替代风险评估（0-100，分数越高越抗AI，即越安全）。
    3. 提供真实的就业薪资估算（基于一线城市水平调整）。
    4. 分析考研与考公的必要性和难易度。
    5. 指出高风险专业和被高估的热门专业。
    6. 行业分布 (industry_distribution) 必须返回一个包含 industry (行业名称) 和 percentage (百分比) 的对象数组。
    
    必须严格按照提供的JSON Schema格式输出。所有货币单位为人民币。
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: responseSchema,
        temperature: 0.3, // Lower temperature for more analytical/factual output
      },
    });

    const text = response.text;
    if (!text) throw new Error("No response from AI");
    
    return JSON.parse(text) as AnalysisResult;
  } catch (error) {
    console.error("Gemini API Error:", error);
    throw error;
  }
};