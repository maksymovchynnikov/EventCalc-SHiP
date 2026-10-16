"""Production envelopes, correlated acceptance and both sensitivity boundaries."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sensitivity import DecayBatch,Geometry,Selection,prepare,ship_selections
from production_model.uncertainties import (FAMILIES,SCALES,PDFWeights,VariationBatch,
                                           pdf_envelope,validate_labels)
from production_bands import (boundary_bands,combine_envelopes,coherent_intervals,
                              integrate_variations)


def labels():
    result=[]
    for family,(count,_) in FAMILIES.items():
        result += [[family,0,*s] for s in SCALES]
        result += [[family,j,1.,1.] for j in range(1,count)]
    return result


class ProductionBandTests(unittest.TestCase):
    def test_pdf_confidence_rules_and_scales(self):
        ll=labels();ix={tuple(x):i for i,x in enumerate(ll)}
        v=np.full((len(ll),2),10.)
        v[ix['CT18NLO',1,1.,1.]]=10+1.6448536269514722*3
        v[ix['CT18NLO',2,1.,1.]]=10-1.6448536269514722*2
        lo,hi=pdf_envelope(v,ll)
        np.testing.assert_allclose(lo,8.);np.testing.assert_allclose(hi,13.)
        v[ix['MSHT20nlo_as118',0,2.,2.]]=20
        self.assertTrue(np.all(pdf_envelope(v,ll)[1]==20))
        for j in range(1,101):v[ix['NNPDF40_nlo_as_01180',j,1.,1.]]=j
        lo,hi=pdf_envelope(v,ll)
        np.testing.assert_allclose(hi,np.quantile(np.arange(1,101),.84))

    def test_correlate_categories_before_pdf_combination(self):
        ll=labels();primary=np.full((len(ll),1),5.);cascade=primary.copy()
        # The same PDF member raises one source and lowers the other.
        primary[7]=9.;cascade[7]=1.;primary[8]=1.;cascade[8]=9.
        lo,hi=pdf_envelope(primary+cascade,ll)
        np.testing.assert_allclose(lo,10);np.testing.assert_allclose(hi,10)
        self.assertGreater(float(pdf_envelope(primary,ll)[1][0]+pdf_envelope(cascade,ll)[1][0]),10)

    def test_missing_or_duplicate_pdf_member_rejected(self):
        for bad in [labels()[:-1],labels()+[labels()[0]]]:
            with self.assertRaises(ValueError):validate_labels(bad)

    def test_missing_package_fails_without_silent_central_fallback(self):
        class Inputs:pass
        with tempfile.TemporaryDirectory() as tmp:
            inputs=Inputs();inputs.root=Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError,'central-only'):PDFWeights(inputs)

    def test_corrupt_weight_file_rejected(self):
        class Inputs:pass
        with tempfile.TemporaryDirectory() as tmp:
            inputs=Inputs();inputs.root=Path(tmp);package=inputs.root/'uncertainties';package.mkdir()
            (package/'labels.json').write_text('[]')
            (package/'manifest.json').write_text(json.dumps({'sha256':{'labels.json':'wrong'}}))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):PDFWeights(inputs)

    def test_stale_source_provenance_rejected(self):
        class Inputs:pass
        with tempfile.TemporaryDirectory() as tmp:
            inputs=Inputs();inputs.root=Path(tmp);package=inputs.root/'uncertainties';package.mkdir()
            source=inputs.root/'results/idm_production/sources';source.mkdir(parents=True)
            (source/'provenance.json').write_text('{"shower":"new"}')
            (package/'source_provenance.json').write_text('{"shower":"old"}')
            (package/'manifest.json').write_text('{"sha256":{}}')
            with self.assertRaisesRegex(ValueError,'do not match'):PDFWeights(inputs)

    def test_chunked_weight_lookup_preserves_event_order(self):
        pdf=object.__new__(PDFWeights);pdf.labels=labels();pdf.rows=5;pdf.stops=[2,5]
        matrix=np.tile(np.arange(5.)[:,None]+2,(1,len(pdf.labels)));matrix[:,0]=1
        pdf.blocks=[matrix[:2],matrix[2:]]
        ids=np.array([4,0,2,1,4]);np.testing.assert_array_equal(pdf.take(ids),matrix[ids])
        with self.assertRaises(ValueError):pdf.take(np.array([5]))

    def test_combination_uses_varied_accepted_yields(self):
        ll=labels();fixed=np.array([[2.,4.],[8.,3.]])
        brem=np.array([[[1.,1.],[1.,1.]],[[.5,.2],[.3,.4]],[[2.,3.],[5.,6.]]])
        dy=np.ones((3,len(ll),2,2));dy[1]*=2;dy[2]*=.5
        central=fixed+brem[0]+dy[0,0]
        lo,hi=combine_envelopes(central,fixed,brem,dy,ll)
        np.testing.assert_allclose(lo,fixed+brem[1]+.5)
        np.testing.assert_allclose(hi,fixed+brem[2]+2)

    def test_both_boundaries_expand_and_contract(self):
        x=np.geomspace(1e-5,1e-1,401);y=20*np.exp(-(np.log(x/1e-3)/1.3)**2)
        result=boundary_bands(x,y,y*.4,y*3)
        c=result['central_intervals'][0];band=result['components'][0]
        lo=band['lower_epsilon2_boundary_range'];hi=band['upper_epsilon2_boundary_range']
        self.assertLess(lo[0],c[0]);self.assertGreater(lo[1],c[0])
        self.assertLess(hi[0],c[1]);self.assertGreater(hi[1],c[1])

    def test_disconnected_and_open_intervals_stay_separate(self):
        x=np.geomspace(1e-9,1e-2,8);y=np.array([5.,5.,0.,0.,5.,5.,0.,5.])
        result=boundary_bands(x,y,y*.9,y*1.1)
        self.assertEqual(len(result['components']),3)
        self.assertIsNone(result['central_intervals'][0][0])
        self.assertIsNone(result['central_intervals'][-1][1])

    def test_disappearing_reach_is_not_given_fabricated_boundaries(self):
        x=np.geomspace(1e-9,1e-2,8);y=np.array([0.,0.,3.,4.,3.,0.,0.,0.])
        b=boundary_bands(x,y,y*.1,y*2)['components'][0]
        self.assertEqual(b['lower_yield_intervals'],[])
        self.assertIsNone(b['lower_epsilon2_boundary_range'])
        self.assertIsNone(b['upper_epsilon2_boundary_range'])

    def test_coherent_variations_extract_both_selection_intervals(self):
        x=np.geomspace(1e-6,1,100);fixed=np.zeros((100,2))
        y=15*np.exp(-(np.log(x/1e-3)/1.5)**2)
        brem=np.stack([np.column_stack([y,y*.8])*f for f in [1.,.5,2.]])
        dy=np.zeros((1,1,100,2));selections=ship_selections()[2:4]
        r=coherent_intervals(x,fixed,brem,dy,[['toy',0,1.,1.]],[2.],selections,2.3)
        self.assertEqual(len(r),3)
        self.assertLess(r[2]['intervals'][selections[0].name][0][0],r[0]['intervals'][selections[0].name][0][0])
        self.assertGreater(r[2]['intervals'][selections[0].name][0][1],r[0]['intervals'][selections[0].name][0][1])

    def test_shared_decay_response_against_direct_varied_integrals(self):
        n=80;m=1.;pz=np.linspace(12.,180.,n);p=np.column_stack([np.zeros((n,2)),pz,np.hypot(pz,m)])
        origins=np.zeros((n,3));origins[:,2]=np.linspace(0,1.25,n)
        # The nominal zero at event 0 acquires a contribution in a variation.
        factors=np.column_stack([np.linspace(0,1,n),np.linspace(2,.2,n),np.full(n,.5)])
        b=VariationBatch(p,origins,np.ones(n)/n,m,'brem',factors)
        d=np.tile([[[.3,0.,.4,.5],[-.3,0.,-.4,.5]]],(n,1,1))
        vis=np.ones((n,2),bool);charge=np.tile([1.,-1.],(n,1))
        def sampler(mass,count,rng):return d,vis,charge,np.ones(count)
        ct=np.geomspace(.01,1e5,151);selections=(ship_selections()[2],ship_selections()[3])
        exact,parts=integrate_variations([b],sampler,ct,selections,seed=1)
        for i in range(3):
            direct=prepare(DecayBatch(p,d,vis,b.weights*factors[:,i],m,origins,frame='rest',charges=charge),geometry=Geometry(),selections=selections).evaluate(ct)
            np.testing.assert_allclose(parts['brem'][i],direct,rtol=0,atol=direct.max()*4e-5)
            if i==0:np.testing.assert_allclose(exact,direct,rtol=1e-12,atol=1e-15)
        # Shape-dependent weighting is measurably different from a constant flux factor.
        ratio=parts['brem'][1,:,0]/np.maximum(parts['brem'][0,:,0],1e-100)
        self.assertGreater(np.ptp(ratio[parts['brem'][0,:,0]>1e-4]),.1)


if __name__=='__main__':unittest.main()
