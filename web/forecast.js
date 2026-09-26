export const names={trend:'Linear trend',seasonal:'Seasonal naive',holt:'Holt’s linear smoothing'};
export function predict(values,horizon,model,period=7){
 if(values.length<3||!values.every(Number.isFinite)||!Number.isInteger(horizon)||horizon<1)throw Error('A valid numeric series and positive horizon are required.');
 const n=values.length;
 if(model==='seasonal'){if(n<period)throw Error('Not enough observations for this season length.');return Array.from({length:horizon},(_,i)=>values[n-period+i%period]);}
 if(model==='trend'){const xm=(n-1)/2,ym=values.reduce((a,b)=>a+b,0)/n;let num=0,den=0;values.forEach((y,x)=>{num+=(x-xm)*(y-ym);den+=(x-xm)**2});const slope=num/den;return Array.from({length:horizon},(_,i)=>ym+slope*(n+i-xm));}
 if(model==='holt'){let level=values[0],trend=values[1]-values[0];for(let i=1;i<n;i++){const previous=level;level=.35*values[i]+.65*(level+trend);trend=.15*(level-previous)+.85*trend;}return Array.from({length:horizon},(_,i)=>level+(i+1)*trend);}
 throw Error('Unknown model.');
}
export function evaluate(values,period=7){const cut=Math.floor(values.length*.8),train=values.slice(0,cut),test=values.slice(cut);return Object.keys(names).filter(m=>m!=='seasonal'||train.length>=period).map(model=>{const p=predict(train,test.length,model,period),errors=test.map((v,i)=>v-p[i]);return {model,mae:errors.reduce((s,v)=>s+Math.abs(v),0)/errors.length,rmse:Math.sqrt(errors.reduce((s,v)=>s+v*v,0)/errors.length),holdout:test.length};}).sort((a,b)=>a.mae-b.mae);}
export function parseCSV(text){
 const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/).filter(x=>x.trim());if(lines.length<22)throw Error('Upload at least 21 observations plus the date,value header.');if(lines.length>5001)throw Error('Please use at most 5,000 observations.');
 const header=lines.shift().toLowerCase().replace(/\s/g,'');if(header!=='date,value')throw Error('The first row must be date,value.');
 const rows=lines.map((line,i)=>{const cells=line.split(',').map(s=>s.trim().replace(/^"|"$/g,''));if(cells.length!==2||!/^\d{4}-\d{2}-\d{2}$/.test(cells[0])||!cells[1])throw Error(`Row ${i+2}: use YYYY-MM-DD and a numeric value.`);const t=Date.parse(cells[0]+'T00:00:00Z'),value=Number(cells[1]);if(!Number.isFinite(t)||new Date(t).toISOString().slice(0,10)!==cells[0]||!Number.isFinite(value))throw Error(`Row ${i+2}: invalid date or value.`);return {date:cells[0],time:t,value};});
 const step=rows[1].time-rows[0].time;if(step<=0)throw Error('Dates must be unique and in increasing order.');if(rows.some((r,i)=>i>0&&r.time-rows[i-1].time!==step))throw Error('Dates must be evenly spaced. Fill missing dates before uploading.');return rows;
}
export function sample(kind){return Array.from({length:84},(_,i)=>{const time=Date.UTC(2026,0,1+i);const wave=Math.sin(i*1.91)*3+Math.cos(i*.63)*2;const value=kind==='demand'?180+i*.7+32*Math.sin(i*2*Math.PI/7)+wave:kind==='energy'?420+85*Math.sin(i*2*Math.PI/7)+wave*4:1200+i*12+140*Math.sin(i*2*Math.PI/7)+wave*10;return {time,date:new Date(time).toISOString().slice(0,10),value:Math.round(value*100)/100};});}
