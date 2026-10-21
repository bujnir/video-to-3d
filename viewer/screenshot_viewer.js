const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined, args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1100, height: 900 } });
  const errs = [];
  p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  p.on('pageerror', e => errs.push(e.message));
  await p.goto('http://127.0.0.1:8765/index.html');
  await p.waitForTimeout(6000);
  const v = await p.$('#view');
  await v.scrollIntoViewIfNeeded();
  await p.waitForTimeout(1500);
  await v.screenshot({ path: process.argv[2] + '/view.png' });
  await p.click('#fly');
  await p.waitForTimeout(4000);
  await v.screenshot({ path: process.argv[2] + '/fly.png' });
  console.log('errors:', JSON.stringify(errs));
  await b.close();
})();
