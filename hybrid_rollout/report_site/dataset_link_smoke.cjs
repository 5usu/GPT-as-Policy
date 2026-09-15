const {chromium}=require(process.env.REPORT_PLAYWRIGHT||'playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const base=process.env.REPORT_URL;
const out=path.resolve(process.env.REPORT_QA||'runtime/report_dataset_link_qa');
const dataset='https://huggingface.co/datasets/YuMoool/astra-robodojo-rollouts';
if(!base)throw new Error('Set REPORT_URL to the built report');
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto(base,{waitUntil:'domcontentloaded'});
  await page.locator('.rr-project-links').waitFor();
  await page.evaluate(()=>document.fonts.ready);
  assert.equal(await page.locator('.rr-report').getAttribute('lang'),'en');
  for(const language of ['en','zh']){
   if(language==='zh')await page.getByRole('button',{name:'Switch to Chinese',exact:true}).click();
   const links=page.locator('.rr-project-links a'),link=page.locator('.rr-project-links .rr-dataset-link');
   assert.equal(await link.count(),1);
   assert.equal(await link.innerText(),language==='en'?'Rollout Records':'评测记录');
   assert.equal(await link.getAttribute('href'),dataset);
   assert.equal(await link.getAttribute('target'),'_blank');
   assert.equal(await link.getAttribute('rel'),'noopener noreferrer');
   assert.equal(await links.nth(0).innerText(),'GitHub');
   assert.equal(await links.nth(1).getAttribute('href'),dataset);
   assert.equal(await links.nth(2).getAttribute('href'),`gallery.html?lang=${language}`);
   assert.equal(await links.nth(3).getAttribute('href'),`robolab-gallery.html?lang=${language}`);
   // Verify exact click destination without an external-site availability dependency.
   await page.context().route(dataset,route=>route.fulfill({status:200,contentType:'text/html',body:'<!doctype html><title>Dataset link check</title>'}));
   const [popup]=await Promise.all([page.waitForEvent('popup'),link.click()]);
   await popup.waitForLoadState('domcontentloaded');assert.equal(popup.url(),dataset);
   await popup.close();await page.context().unroute(dataset);
   for(const width of [1440,390]){
    await page.setViewportSize({width,height:width===390?844:1050});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2));
    const boxes=await links.evaluateAll(xs=>xs.map(x=>{const r=x.getBoundingClientRect();return{x:r.x,right:r.right,width:r.width,height:r.height};}));
    assert(boxes.every(r=>r.width>0&&r.height>0&&r.x>=0&&r.right<=width+2));
    await page.locator('.rr-hero').screenshot({path:path.join(out,`dataset-${language}-${width}.png`)});
   }
   await page.setViewportSize({width:1440,height:1050});
  }
  assert.equal(await page.locator('.rr-author').count(),6);
  assert.equal(await page.locator('.rr-clip').count(),21);
  assert.deepEqual(errors,[]);
  const result={status:'passed',dataset,clicks:2,bilingual:true,desktop:true,mobile:true,existingGalleryLinks:true,authors:6,bodyClips:21,pageErrors:errors};
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
