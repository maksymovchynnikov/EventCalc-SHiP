"""One-mass perfect-reconstruction scans with the calibrated production default.

This adapter supplies the DP/iDM coupling laws. sensitivity.py remains model
independent. Decays, production and acceptance stay in memory; no HepMC step.
"""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
from sensitivity import integrate,ship_selections,threshold_intervals,charges_from_pdg,Geometry
from production_model import ProductionModel
from production_model.uncertainties import ProductionVariations,FORM_FACTORS,OMITTED
from production_bands import integrate_variations,combine_envelopes,boundary_bands,coherent_intervals

def configure_exhad(root=None):
    root=root or os.environ.get('EXHAD_ROOT') or os.environ.get('DP_EXHAD_ROOT')
    if root:
        path=str(Path(root).expanduser().resolve())
        existing=sys.modules.get('exhad')
        if existing is not None and str(Path(existing.__file__).resolve().parents[1])!=path:
            raise RuntimeError('A different exHad release is already imported')
        if path not in sys.path:sys.path.insert(0,path)
    try:import exhad
    except ImportError as error:raise ImportError('Install exHad or supply --exhad-root / EXHAD_ROOT') from error
    return exhad

def decay_library(model,mass,delta,count,seed):
    from exhad.inelastic_dark_matter import width as idm_width,generate,HBARC
    if model=='idm':
        result=generate(mass,count,delta=delta,seed=seed);events=result['particles'];width=idm_width(mass,delta=delta)
    else:
        if 3.05<=mass<=3.14 or 3.64<=mass<=3.73:raise ValueError('The decay model excludes narrow charmonium intervals')
        if mass>5:
            from production_model.continuum_decays import generate as continuum
            events,width,_=continuum(mass,count,seed)
        else:
            from exhad.api import Generator
            from production_model.spectral_widths import spectral_point
            with Generator('dark-photon',workers=1,chunk_size=2048) as generator:
                result=generator.generate_all(mass,count,seed=seed)
            events=result['events'];width=float(spectral_point(mass).sum())
    a=np.zeros((len(events),max(map(len,events)),6));a[:,:,5]=-999
    for i,event in enumerate(events):a[i,:len(event)]=event
    visible=~np.isin(np.abs(a[:,:,5]),[999,12,14,16,4900101])
    charges=charges_from_pdg(a[:,:,5],visible=visible)
    def sampler(mother_mass,n,rng):
        ix=rng.integers(len(a),size=n);d=a[ix,:,:4].copy()
        q=rng.normal(size=(n,4));q/=np.linalg.norm(q,axis=1)[:,None];x,y,z,w=q.T
        rot=np.empty((n,3,3));rot[:,0,0]=1-2*(y*y+z*z);rot[:,0,1]=2*(x*y-z*w);rot[:,0,2]=2*(x*z+y*w)
        rot[:,1,0]=2*(x*y+z*w);rot[:,1,1]=1-2*(x*x+z*z);rot[:,1,2]=2*(y*z-x*w)
        rot[:,2,0]=2*(x*z-y*w);rot[:,2,1]=2*(y*z+x*w);rot[:,2,2]=1-2*(x*x+y*y)
        d[:,:,:3]=np.einsum('nij,nkj->nki',rot,d[:,:,:3])
        return d,visible[ix],charges[ix],np.ones(n)
    return sampler,HBARC/width

def scan_mass(model,mass,epsilon2,*,delta=.1,power=15,decay_count=32768,replicas=2,seed=12015,exposure=6e20,exhad_root=None,data_root=None,response_nodes=8192,production_uncertainties=True,pdf_weights=None,threshold_events=2.3):
    configure_exhad(exhad_root);eps=np.asarray(epsilon2,float)
    if eps.ndim!=1 or len(eps)<2 or np.any(~np.isfinite(eps)) or np.any(eps<=0) or np.any(np.diff(eps)<=0):raise ValueError('epsilon2 must be a strictly increasing positive grid')
    if model not in ('dp','idm') or replicas<2:raise ValueError('Choose dp/idm and at least two integration replicas')
    if not np.isfinite(mass) or mass<=0 or not np.isfinite(delta) or delta<=0:raise ValueError('Mass and splitting must be positive')
    if decay_count<1 or not np.isfinite(exposure) or exposure<=0:raise ValueError('Positive decay count and exposure required')
    if not np.isfinite(threshold_events) or threshold_events<=0:raise ValueError('Positive event threshold required')
    if isinstance(power,bool) or int(power)!=power or not 4<=power<=24:raise ValueError('Integration power must be an integer from 4 to 24')
    if not isinstance(response_nodes,int) or response_nodes<2:raise ValueError('At least two response nodes required')
    closed=model=='idm' and mass*delta/3<=2*.00051099895
    started=time.perf_counter();production=ProductionModel(data_root)
    variations=ProductionVariations(production,pdf_weights) if production_uncertainties else None
    selection=(ship_selections()[2],ship_selections()[3]);results=[];components=[]
    endpoints=variations.endpoints(model,mass,delta) if variations else ()
    interpolation_error=0.
    ctau_unit=np.inf
    for replica in range(replicas):
        if closed:
            results.append(np.zeros((len(eps),len(selection))))
            if variations:components.append({})
            continue
        key=seed+replica*100003;sampler,ctau_unit=decay_library(model,mass,delta,decay_count,key)
        if variations:
            batches=variations.batches(mass,model=model,delta=delta,power=power,seed=key)
            exact,parts=integrate_variations(batches,sampler,ctau_unit/eps,selection,seed=key,nodes=response_nodes)
            interpolated=sum((part[0] for part in parts.values()),np.zeros_like(exact))
            # Check the response where it could influence either threshold crossing.
            scale=exposure*eps[:,None];relevant=(exact*scale>.01)&(exact>exact.max(initial=0)*1e-12)
            if relevant.any():interpolation_error=max(interpolation_error,float(np.max(np.abs(interpolated[relevant]-exact[relevant])/exact[relevant])))
            if interpolation_error>.003:
                raise ValueError(f'Lifetime-response interpolation error {interpolation_error:.3g}; increase --response-nodes')
            results.append(exact*scale);components.append({group:values*scale[None,:,:] for group,values in parts.items()})
        else:
            batches=production.decay_batches(mass,sampler,model=model,delta=delta,power=power,seed=key)
            results.append(integrate(batches,ctau_unit/eps,selections=selection,exposure=exposure,production_scales=eps,response_nodes=response_nodes))
    results=np.array(results);events=results.mean(0);error=results.std(0,ddof=1)/np.sqrt(replicas)
    arrays={'epsilon2':eps,'ctau_m':ctau_unit/eps,'events':events,'integration_error_events':error}
    metadata={'schema':2,'model':model,'mediator_mass_GeV':mass,'delta':delta,'ratio':3,'alpha_D':.1,'protons_on_target':exposure,'threshold_events':threshold_events,'response_nodes':response_nodes,'geometry':Geometry().__dict__,'electron_channel_closed':closed,'production':production.metadata,'reconstruction_efficiency':1.,'selections':[s.__dict__ for s in selection],'intervals':{s.name:threshold_intervals(eps,events[:,i],threshold_events) for i,s in enumerate(selection)},'production_uncertainties':bool(production_uncertainties),'integration_error_scope':'Independent integration replicas, conditional on saved production pools; separate from production uncertainty.'}
    if variations:
        nv=len(variations.pdf.labels);ne=len(endpoints);shape=(len(eps),len(selection))
        counts={'fixed':1,'brem':3,'dy':ne*nv}
        parts={group:np.array([p.get(group,np.zeros((count,*shape))) for p in components]) for group,count in counts.items()}
        fixed=parts['fixed'].mean(axis=0)[0];brem=parts['brem'].mean(axis=0)
        dy=parts['dy'].mean(axis=0).reshape(ne,nv,*shape)
        lower,upper=combine_envelopes(events,fixed,brem,dy,variations.pdf.labels)
        arrays.update(lower_events=lower,upper_events=upper,fixed_events=fixed,
                      brem_variation_events=brem,dy_variation_events=dy,
                      brem_variation_integration_error_events=parts['brem'].std(axis=0,ddof=1)/np.sqrt(replicas),
                      dy_variation_integration_error_events=(parts['dy'].std(axis=0,ddof=1)/np.sqrt(replicas)).reshape(ne,nv,*shape))
        metadata.update(pdf_labels=variations.pdf.labels,form_factors=list(FORM_FACTORS),
            transition_endpoints_GeV=list(endpoints),uncertainty_data=variations.pdf.manifest,
            response_max_relative_error=interpolation_error,
            uncertainty_scope='Accepted-yield envelope of CT18/MSHT20 Hessian and NNPDF replica intervals, seven scales per family, three form-factor pairs at fixed fitted Lambda_p, and DY transition endpoints 1.75–2.5 GeV. Common variations across primary/cascade production and the iDM virtual-mass integral; not a joint statistical confidence interval.',
            omitted_uncertainties=OMITTED,
            boundary_bands={s.name:boundary_bands(eps,events[:,i],lower[:,i],upper[:,i],threshold_events) for i,s in enumerate(selection)},
            variation_intervals=coherent_intervals(eps,fixed,brem,dy,variations.pdf.labels,endpoints,selection,threshold_events),
            boundary_convention='All intervals are in epsilon squared. Both lower and upper crossings are extracted separately. Null interval endpoints mark open scan boundaries; split or disappearing components are not joined.')
    else:metadata['uncertainty_scope']='Central-only scan explicitly requested; integration errors only, production bands not calculated.'
    if not all(np.all(np.isfinite(v)) for k,v in arrays.items() if k!='ctau_m'):
        raise ValueError('Nonfinite event yields or production bands')
    metadata['seconds']=time.perf_counter()-started
    return arrays,metadata

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['dp','idm'],required=True);p.add_argument('--mass',type=float,required=True);p.add_argument('--delta',type=float,default=.1);p.add_argument('--epsilon2-min',type=float,default=1e-20);p.add_argument('--epsilon2-max',type=float,default=.1);p.add_argument('--points',type=int,default=401);p.add_argument('--power',type=int,default=15);p.add_argument('--replicas',type=int,default=2);p.add_argument('--exhad-root');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--central-only',action='store_true',help='Explicitly skip production bands (default: calculate them)')
    p.add_argument('--pdf-weights',type=Path,help='Override the bundled, source-matched uncertainty package')
    p.add_argument('--data-root',type=Path);p.add_argument('--decay-count',type=int,default=32768)
    p.add_argument('--response-nodes',type=int,default=8192);p.add_argument('--seed',type=int,default=12015)
    a=p.parse_args()
    if a.output.exists():p.error('Choose a new output directory')
    arrays,metadata=scan_mass(a.model,a.mass,np.geomspace(a.epsilon2_min,a.epsilon2_max,a.points),delta=a.delta,power=a.power,replicas=a.replicas,exhad_root=a.exhad_root,data_root=a.data_root,decay_count=a.decay_count,response_nodes=a.response_nodes,seed=a.seed,production_uncertainties=not a.central_only,pdf_weights=a.pdf_weights)
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'events.npz',**arrays);(a.output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n');print(json.dumps({'output':str(a.output),'seconds':metadata['seconds'],'production_uncertainties':metadata['production_uncertainties'],'boundary_bands':metadata.get('boundary_bands')}))
if __name__=='__main__':main()
