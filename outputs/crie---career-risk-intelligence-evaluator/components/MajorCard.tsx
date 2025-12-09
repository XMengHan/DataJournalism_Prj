import React, { useState } from 'react';
import { MajorRecommendation } from '../types';
import { SalaryChart, IndustryPieChart } from './Charts';
import { ChevronDown, ChevronUp, MapPin, Briefcase, GraduationCap, ShieldAlert, Cpu } from 'lucide-react';

interface Props {
  data: MajorRecommendation;
  index: number;
}

export const MajorCard: React.FC<Props> = ({ data, index }) => {
  const [expanded, setExpanded] = useState(index === 0);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden mb-6 transition-all duration-300 hover:shadow-md">
      {/* Header Section */}
      <div 
        className="p-6 cursor-pointer bg-gradient-to-r from-white to-gray-50"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex justify-between items-start">
          <div className="flex items-center gap-4">
            <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary/10 text-primary font-serif font-bold text-xl">
              #{data.rank}
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                {data.major_name}
                <span className="text-xs font-normal px-2 py-0.5 bg-gray-100 text-gray-500 rounded-full">
                  {data.major_code}
                </span>
                <span className="text-xs font-normal px-2 py-0.5 bg-blue-50 text-blue-600 rounded-full border border-blue-100">
                  {data.category}
                </span>
              </h3>
              <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
                <span className="flex items-center gap-1">
                  <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary rounded-full" 
                      style={{ width: `${data.match_score}%` }}
                    />
                  </div>
                  <span className="font-medium text-primary">{data.match_score}% 匹配</span>
                </span>
              </div>
            </div>
          </div>
          
          <div className="text-right hidden sm:block">
            <div className="text-sm text-gray-500">抗AI指数</div>
            <div 
              className="font-bold text-lg" 
              style={{ color: data.ai_resistance_index.color }}
            >
              {data.ai_resistance_index.score}/100
            </div>
          </div>
          
          <button className="sm:hidden text-gray-400">
             {expanded ? <ChevronUp /> : <ChevronDown />}
          </button>
        </div>
        
        {/* Quick Tags - Always Visible on Desktop */}
        <div className="mt-4 flex flex-wrap gap-2">
           {data.match_reasons.slice(0, 2).map((reason, i) => (
             <span key={i} className="text-xs bg-green-50 text-green-700 px-2 py-1 rounded border border-green-100">
               ✓ {reason}
             </span>
           ))}
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div className="p-6 border-t border-gray-100 bg-white animate-fadeIn">
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Left Column: Employment & AI */}
            <div className="space-y-6">
              
              {/* Employment Stats */}
              <div>
                <h4 className="flex items-center gap-2 font-bold text-gray-800 mb-3">
                  <Briefcase size={18} className="text-primary" /> 就业大数据
                </h4>
                <div className="grid grid-cols-2 gap-4 bg-gray-50 p-4 rounded-lg">
                  <div>
                    <div className="text-xs text-gray-500">就业率</div>
                    <div className="text-lg font-bold text-gray-900">{data.employment_data.employment_rate}%</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">薪资增长潜力</div>
                    <div className="text-lg font-bold text-success">{data.employment_data.salary_growth_rate}%</div>
                  </div>
                </div>
                <div className="mt-4">
                  <div className="text-xs text-gray-500 mb-1">本硕薪资对比 (月薪)</div>
                  <SalaryChart 
                    bachelor={data.employment_data.avg_salary_bachelor}
                    master={data.employment_data.avg_salary_master}
                  />
                </div>
              </div>

              {/* AI Analysis */}
              <div className="bg-green-50/50 p-4 rounded-lg border border-green-100">
                <h4 className="flex items-center gap-2 font-bold text-gray-800 mb-2">
                  <Cpu size={18} className="text-success" /> AI 风险智能评估
                </h4>
                <p className="text-sm text-gray-600 mb-2 font-medium">
                  {data.ai_resistance_index.level} - {data.ai_resistance_index.interpretation}
                </p>
                <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                  <div 
                    className="bg-success h-2 rounded-full transition-all duration-1000" 
                    style={{ width: `${data.ai_resistance_index.score}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500">{data.ai_resistance_index.future_outlook}</p>
              </div>

            </div>

            {/* Right Column: Education & Path */}
            <div className="space-y-6">
              
               {/* University & Exam */}
              <div>
                <h4 className="flex items-center gap-2 font-bold text-gray-800 mb-3">
                  <GraduationCap size={18} className="text-primary" /> 升学与发展
                </h4>
                
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">考研推荐度</span>
                    <span className="font-bold text-primary">{data.graduate_public_exam.graduate_roi}</span>
                  </div>
                  <p className="text-xs text-gray-500">{data.graduate_public_exam.graduate_school_recommendation}</p>
                </div>

                <div className="mb-4">
                   <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">考公岗位量</span>
                    <span className="font-bold text-gray-900">{data.graduate_public_exam.civil_service_positions}+</span>
                  </div>
                  <p className="text-xs text-gray-500">{data.graduate_public_exam.civil_service_recommendation}</p>
                </div>
                
                 <div className="bg-gray-50 p-3 rounded-lg">
                  <div className="text-xs font-bold text-gray-700 mb-2">推荐院校 (基于分数)</div>
                  <div className="space-y-2">
                    {data.suggested_universities.map((uni, i) => (
                      <div key={i} className="flex justify-between items-center text-sm">
                        <span className="flex-1 text-gray-800">{uni.name}</span>
                        <span className="text-xs bg-white border px-1 rounded text-gray-500 mr-2">{uni.ranking}</span>
                        <span className={`text-xs font-bold ${uni.admission_probability > 80 ? 'text-green-600' : 'text-orange-500'}`}>
                          {uni.admission_probability}% 概率
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

               {/* Industry Dist */}
               <div>
                  <h4 className="font-bold text-gray-800 mb-2 text-sm">行业分布</h4>
                  <IndustryPieChart distribution={data.employment_data.industry_distribution} />
               </div>

            </div>
          </div>

          {/* Warnings Footer */}
          {data.warnings.length > 0 && (
            <div className="mt-6 bg-orange-50 p-3 rounded-lg border border-orange-100 flex gap-3 items-start">
              <ShieldAlert size={18} className="text-orange-500 shrink-0 mt-0.5" />
              <div className="text-sm text-orange-800">
                <span className="font-bold block mb-1">风险提示</span>
                <ul className="list-disc pl-4 space-y-1">
                  {data.warnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
