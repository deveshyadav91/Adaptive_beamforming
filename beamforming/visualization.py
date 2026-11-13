import numpy as np
import matplotlib.pyplot as plt
from .array import steering_vector, db20
from .impairments import cosine_power_element_pattern

def planar_pattern_grid(positions_lambda, weights, theta_deg=None, phi_deg=None, element_exponent=0.0):
    theta = np.linspace(0,180,91) if theta_deg is None else np.asarray(theta_deg)
    phi = np.linspace(-180,180,181) if phi_deg is None else np.asarray(phi_deg)
    mag = np.empty((len(theta),len(phi)))
    for i,t in enumerate(theta):
        for j,p in enumerate(phi):
            mag[i,j] = abs(np.vdot(weights, steering_vector(positions_lambda,t,p)))
        mag[i,:] *= cosine_power_element_pattern(t, element_exponent)
    mag /= max(float(mag.max()),1e-15)
    return theta, phi, mag, db20(mag)

def plot_azimuth_elevation_heatmap(positions_lambda, weights, floor_db=-40, element_exponent=0.0):
    az = np.linspace(-180,180,181)
    el = np.linspace(-90,90,91)
    mag = np.empty((len(el),len(az)))
    for i,e in enumerate(el):
        for j,a in enumerate(az):
            mag[i,j] = abs(np.vdot(weights, steering_vector(positions_lambda,90-e,a)))
        mag[i,:] *= cosine_power_element_pattern(90-e, element_exponent)
    mag /= max(float(mag.max()),1e-15)
    pat = np.maximum(db20(mag),floor_db)
    fig,ax = plt.subplots(figsize=(10,4.8))
    im=ax.imshow(pat,origin="lower",aspect="auto",extent=[az[0],az[-1],el[0],el[-1]],vmin=floor_db,vmax=0)
    ax.set(xlabel="Azimuth (degrees)",ylabel="Elevation (degrees)",title="UPA azimuth/elevation response")
    fig.colorbar(im,ax=ax,label="Normalized response (dB)")
    fig.tight_layout()
    return fig

def plot_3d_pattern(positions_lambda, weights, floor_db=-35, element_exponent=0.0):
    theta,phi,mag,pat=planar_pattern_grid(positions_lambda,weights,element_exponent=element_exponent)
    th,ph=np.meshgrid(np.deg2rad(theta),np.deg2rad(phi),indexing="ij")
    r=np.maximum(mag,10**(floor_db/20))
    x=r*np.sin(th)*np.cos(ph); y=r*np.sin(th)*np.sin(ph); z=r*np.cos(th)
    fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection="3d")
    ax.plot_surface(x,y,z,rstride=2,cstride=2,linewidth=0,antialiased=True)
    ax.set(xlabel="x",ylabel="y",zlabel="z",title="Normalized 3D array pattern")
    ax.set_box_aspect((1,1,1))
    return fig
