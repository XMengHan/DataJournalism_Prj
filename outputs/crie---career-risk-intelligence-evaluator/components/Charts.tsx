import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';
import { IndustryDistributionItem } from '../types';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

interface SalaryChartProps {
  bachelor: number;
  master: number;
}

export const SalaryChart: React.FC<SalaryChartProps> = ({ bachelor, master }) => {
  const data = [
    { name: '本科起薪', salary: bachelor },
    { name: '硕士起薪', salary: master },
  ];

  return (
    <div className="h-48 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" hide />
          <YAxis dataKey="name" type="category" width={60} tick={{fontSize: 12}} />
          <Tooltip 
            formatter={(value: number) => `¥${value.toLocaleString()}`}
            contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #eee' }}
          />
          <Bar dataKey="salary" radius={[0, 4, 4, 0]} barSize={20}>
            {data.map((entry, index) => (
               <Cell key={`cell-${index}`} fill={index === 0 ? '#3498db' : '#9b59b6'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

interface IndustryPieChartProps {
  distribution: IndustryDistributionItem[];
}

export const IndustryPieChart: React.FC<IndustryPieChartProps> = ({ distribution }) => {
  const data = distribution.map((item) => ({ name: item.industry, value: item.percentage }));

  return (
    <div className="h-48 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={40}
            outerRadius={60}
            paddingAngle={5}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap justify-center gap-2 text-xs text-gray-500 mt-2">
        {data.slice(0, 4).map((entry, index) => (
          <div key={index} className="flex items-center">
            <div className="w-2 h-2 rounded-full mr-1" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
            {entry.name} {entry.value}%
          </div>
        ))}
      </div>
    </div>
  );
};