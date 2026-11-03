// k6 load test for GET /recommend/{user_id}.
//
//   python scripts/k6_users.py > loadtest/users.json
//   k6 run loadtest/recommend.js                       # against localhost:8000
//   k6 run -e BASE_URL=http://host:8000 -e RATE=200 loadtest/recommend.js
import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';

const users = new SharedArray('users', () => JSON.parse(open('./users.json')));
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  scenarios: {
    steady: {
      executor: 'constant-arrival-rate',
      rate: Number(__ENV.RATE || 100),       // requests per second
      timeUnit: '1s',
      duration: __ENV.DURATION || '60s',
      preAllocatedVUs: 50,
      maxVUs: 200,
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.001'],
    http_req_duration: ['p(99)<100'],
  },
  summaryTrendStats: ['avg', 'p(50)', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

export default function () {
  const user = users[Math.floor(Math.random() * users.length)];
  const res = http.get(`${BASE_URL}/recommend/${user}?k=10`, { tags: { name: 'recommend' } });
  check(res, {
    'status 200': (r) => r.status === 200,
    '10 items': (r) => r.json('items').length === 10,
  });
}
