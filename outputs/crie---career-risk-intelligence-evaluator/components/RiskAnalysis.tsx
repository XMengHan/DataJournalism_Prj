import React from 'react';
import { RiskMajor, OverratedMajor } from '../types';
import { AlertTriangle, TrendingDown } from 'lucide-react';

interface Props {
  highRisk: RiskMajor[];
  overrated: OverratedMajor[];
}

export const RiskAnalysis: React.FC<Props> = ({ highRisk, overrated }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
      {/* High Risk Majors */}
      <div className="bg-white rounded-xl shadow-sm border border-red-100 overflow-hidden">
        <div className="bg-red-50 p-4 border-b border-red-100 flex items-center gap-2">
          <AlertTriangle className="text-danger" size={20} />
          <h3 className="font-bold text-red-900">高风险专业警示</h3>
        </div>
        <div className="p-4">
          {highRisk.map((item, idx) => (
            <div key={idx} className="mb-4 last:mb-0 pb-4 last:pb-0 border-b last:border-0 border-red-50">
              <div className="flex justify-between items-center mb-1">
                <span className="font-bold text-gray-800">{item.major}</span>
                <span className="text-xs px-2 py-0.5 bg-red-100 text-red-600 rounded">AI风险: {item.ai_risk}%</span>
              </div>
              <p className="text-sm text-gray-600">{item.reason}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Overrated Majors */}
      <div className="bg-white rounded-xl shadow-sm border border-orange-100 overflow-hidden">
        <div className="bg-orange-50 p-4 border-b border-orange-100 flex items-center gap-2">
          <TrendingDown className="text-orange-600" size={20} />
          <h3 className="font-bold text-orange-900">被高估/需避雷专业</h3>
        </div>
        <div className="p-4">
          {overrated.map((item, idx) => (
            <div key={idx} className="mb-4 last:mb-0 pb-4 last:pb-0 border-b last:border-0 border-orange-50">
              <div className="flex justify-between items-center mb-1">
                <span className="font-bold text-gray-800">{item.major}</span>
                <span className="text-xs text-orange-500">热度: {item.social_popularity}</span>
              </div>
              <p className="text-sm text-gray-600 mb-1">Reality: {item.reality_check}</p>
              <p className="text-xs text-gray-500 italic">💡 建议: {item.suggestion}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
