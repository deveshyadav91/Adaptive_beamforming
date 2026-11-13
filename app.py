import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from beamforming.array import ula_positions, upa_positions, steering_vector
from beamforming.adaptive import conventional_weights, mvdr_weights, lms_train
from beamforming.simulation import simulate_snapshots
from beamforming.metrics import (pattern_db,hpbw_deg_robust,peak_sidelobe_level_db,
    null_depth_db,visible_grating_lobes_ula,output_sinr_db,main_lobe_peak_index,peak_sidelobe_level_null_based)
from beamforming.visualization import plot_azimuth_elevation_heatmap,plot_3d_pattern
from beamforming.impairments import ArrayModel
from beamforming.robustness import Scenario
import robustness_ui

st.set_page_config(page_title="Adaptive Beamforming Engine",layout="wide")
st.title("Adaptive Digital Beamforming & Null-Steering Engine")
st.caption("Beamforming, null steering, grating-lobe analysis and robustness to real-world array effects (the first three tabs assume an ideal array)")

with st.sidebar:
    st.header("ULA scenario")
    n=st.slider("Number of elements",4,32,12)
    d=st.slider("Element spacing (λ)",0.25,1.25,0.5,0.05)
    target=st.slider("Desired angle from broadside (°)",-80,80,10)
    i1=st.slider("Interferer 1 angle (°)",-80,80,-35)
    second=st.checkbox("Enable second interferer",True)
    i2=st.slider("Interferer 2 angle (°)",-80,80,50,disabled=not second)
    pdes=st.number_input("Desired power",0.1,100.0,1.0)
    pint=st.number_input("Interference power (each)",0.1,1000.0,10.0)
    noise=st.number_input("Noise power per element",0.001,10.0,0.1)
    snapshots=st.select_slider("Snapshots",options=[256,512,1024,2048,4096,8192],value=4096)
    loading=st.select_slider("MVDR diagonal loading",options=[1e-5,1e-4,1e-3,1e-2,1e-1],value=1e-3)
    mu=st.number_input("LMS step size μ",0.00001,0.1,0.001,format="%.5f")
    epochs=st.slider("LMS epochs",1,20,3)

taper_name,kaiser_beta,imp,cal_seed=robustness_ui.sidebar_controls()

pos=ula_positions(n,d)
interferers=[i1]+([i2] if second else [])
powers=[float(pint)]*len(interferers)
X,desired_ref,_=simulate_snapshots(pos,target,interferers,int(snapshots),float(pdes),powers,float(noise),seed=7)
a=steering_vector(pos,target)
weights={"Conventional":conventional_weights(a),
         "MVDR":mvdr_weights(X,a,float(loading))}
weights["LMS"],mse=lms_train(X,desired_ref,float(mu),int(epochs),weights["Conventional"])
angles=np.linspace(-90,90,1801)
patterns={k:pattern_db(pos,w,angles) for k,w in weights.items()}
sinrs={k:output_sinr_db(w,target,interferers,float(pdes),powers,float(noise),pos) for k,w in weights.items()}

scn=Scenario(n_elements=n,spacing_lambda=d,target_deg=float(target),interferer_deg=tuple(interferers),
             desired_power=float(pdes),interference_power=float(pint),noise_power=float(noise),
             n_snapshots=int(snapshots),mvdr_loading=float(loading),lms_mu=float(mu),lms_epochs=int(epochs),
             taper=taper_name,kaiser_beta=kaiser_beta)
try:
    ArrayModel(pos,imp)   # validate impairment settings (e.g. coupling strength)
except ValueError as exc:
    st.sidebar.error(str(exc)); st.stop()

tabs=st.tabs(["Patterns & metrics","2D planar scan","Spacing sweep","Tapers & element pattern",
              "Ideal vs non-ideal","Robustness sweeps","Monte Carlo","Assumptions"])
with tabs[0]:
    c1,c2=st.columns([1.5,1])
    with c1:
        fig,ax=plt.subplots(figsize=(10,5))
        for name,pat in patterns.items(): ax.plot(angles,pat,label=name)
        ax.axvline(target,color="green",ls="--",label="Desired")
        for j,ang in enumerate(interferers): ax.axvline(ang,color="red",ls=":",label="Interferer" if j==0 else None)
        ax.set(xlim=(-90,90),ylim=(-60,2),xlabel="Angle from broadside (°)",ylabel="Normalized response (dB)")
        ax.grid(alpha=.25);ax.legend();fig.tight_layout();st.pyplot(fig);plt.close(fig)
    with c2:
        rows=[]
        for name,pat in patterns.items():
            width=hpbw_deg_robust(angles,pat,target)
            psl=peak_sidelobe_level_null_based(angles,pat,main_lobe_peak_index(angles,pat,target))
            rows.append({"Beamformer":name,"HPBW (°)":width,"Peak SLL (dB)":psl,"Output SINR (dB)":sinrs[name]})
        st.subheader("Quantitative comparison")
        st.dataframe(pd.DataFrame(rows).round(2),hide_index=True,use_container_width=True)
        st.caption("Peak sidelobe level is measured outside the main lobe's first nulls; a grating lobe shows up as about 0 dB.")
        fig,ax=plt.subplots();ax.plot(np.arange(1,len(mse)+1),10*np.log10(np.maximum(mse,1e-15)),marker="o")
        ax.set(xlabel="Epoch",ylabel="MSE (dB)",title="LMS training error");ax.grid(alpha=.3);st.pyplot(fig);plt.close(fig)
    nulls=[{"Beamformer":name,"Interferer (°)":ang,"Local minimum (dB)":null_depth_db(angles,pat,ang,1)}
           for name,pat in patterns.items() for ang in interferers]
    st.subheader("Interference-direction response")
    st.dataframe(pd.DataFrame(nulls).round(2),hide_index=True,use_container_width=True)
    gl=visible_grating_lobes_ula(d,target)
    st.subheader("Theoretical grating-lobe prediction")
    if gl: st.warning("Visible grating-lobe orders predicted by the ideal ULA equation.");st.dataframe(pd.DataFrame(gl),hide_index=True)
    else: st.success("No additional visible grating-lobe orders predicted for this scan.")

with tabs[1]:
    st.subheader("Planar-array directional scan")
    nx=st.slider("UPA x-elements",2,12,6);ny=st.slider("UPA y-elements",2,12,6)
    ds=st.slider("UPA spacing (λ)",0.25,1.0,0.5,0.05)
    upa_el=st.slider("UPA element-pattern exponent p (0 = isotropic)",0.0,3.0,0.0,0.1)
    theta0=st.slider("Steering polar angle θ (°)",5,175,35)
    phi0=st.slider("Steering azimuth φ (°)",-180,180,30)
    upos=upa_positions(nx,ny,ds);uw=conventional_weights(steering_vector(upos,theta0,phi0))
    st.pyplot(plot_azimuth_elevation_heatmap(upos,uw,element_exponent=upa_el))
    if st.checkbox("Also display 3D pattern"): st.pyplot(plot_3d_pattern(upos,uw,element_exponent=upa_el))
    st.caption("This planar-array tab currently demonstrates conventional steering, not adaptive UPA nulling.")

with tabs[2]:
    st.subheader("Element-spacing sweep")
    spacings=st.multiselect("Spacing values (λ)",[0.25,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2],default=[0.3,0.5,0.7,1.0])
    if st.button("Run sweep",type="primary"):
        rows=[]
        for ds in spacings:
            pp=ula_positions(n,ds);ww=conventional_weights(steering_vector(pp,target));pat=pattern_db(pp,ww,angles)
            bw=hpbw_deg_robust(angles,pat,target)
            sll=peak_sidelobe_level_null_based(angles,pat,main_lobe_peak_index(angles,pat,target))
            rows.append({"Spacing (λ)":ds,"HPBW (°)":bw,"Peak SLL (dB)":sll,"Predicted grating lobes":len(visible_grating_lobes_ula(ds,target))})
        df=pd.DataFrame(rows);st.dataframe(df.round(3),hide_index=True,use_container_width=True)
        st.download_button("Download CSV",df.to_csv(index=False).encode(), "spacing_sweep.csv","text/csv")
with tabs[3]:
    robustness_ui.render_taper_tab(scn,imp)
with tabs[4]:
    robustness_ui.render_compare_tab(scn,imp,cal_seed)
with tabs[5]:
    robustness_ui.render_sweep_tab(scn,imp,cal_seed)
with tabs[6]:
    robustness_ui.render_mc_tab(scn,imp,cal_seed)
with tabs[7]:
    st.markdown("""
    **The first three tabs** assume narrowband far-field plane waves, ideal isotropic elements, independent sources,
    spatially white complex Gaussian noise and perfect calibration. LMS uses a clean desired reference (simulation-only).

    **Non-ideal array model:** actual response `v = Γ · M · (e(θ) · a(θ))` with cosine-power element field pattern `e`,
    nearest/next-neighbour coupling `M` (symmetric banded Toeplitz, user-set complex coefficients), and random
    per-channel gain (log-normal, σ in dB) and phase (Gaussian, σ in degrees) errors `Γ` after coupling.
    Receiver noise is added after `Γ` and stays spatially white. Beamformers are designed with the *nominal* steering
    vector, i.e. the processor does not know the errors. Tapers apply to the conventional beamformer (and the LMS initial
    weights); MVDR is not tapered. The coupling model is a simplification, not a full-wave EM solution; the spacing
    scaling option is a heuristic.

    **Not modelled:** polarization, near-field effects, wideband beam squint, noise coloured by coupling,
    nonlinear RF effects, time-varying calibration, and adaptive UPA nulling.
    """)
