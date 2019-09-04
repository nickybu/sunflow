package org.sunflow.core.shader;

import com.nicky.Spectrum;
import com.nicky.brdfs.BRDF;
import javafx.util.Pair;
import org.joml.Vector3f;
import org.sunflow.SunflowAPI;
import org.sunflow.core.*;
import org.sunflow.image.Color;
import org.sunflow.math.OrthoNormalBasis;
import org.sunflow.math.Vector3;

/**
 * <h1>Nicky Shader</h1>
 * Adapter for the BRDF Framework to plug in custom BRDFs
 * Supports diffuse and specular components
 *
 * @author Nicky Buttigieg
 * @version 1.0
 * @since 2018-05-23
 */
public class BRDFFrameworkShader implements Shader {
    private BRDF brdf;
    private final int numRays = 24;

    public BRDFFrameworkShader() {
        this.brdf = null;
    }

    public BRDFFrameworkShader(BRDF brdf) {
        this.brdf = brdf;
    }

    public boolean update(ParameterList pl, SunflowAPI api) {
        brdf = pl.getBRDF("nicky_shader", brdf);
        return true;
    }

    public BRDF getBrdf() {
        return brdf;
    }

    /**
     * Implements Backwards Ray Tracing
     *
     * @param state current render state
     * @return Color The color at that point on the surface
     */
    public Color getRadiance(ShadingState state) {
        // make sure we are on the right side of the material
        state.faceforward();
        // direct lighting
        state.initLightSamples();
        state.initCausticSamples();

        // Incident direction
        Vector3 inDir = state.getRay().getDirection();

        // Get exitant direction and reflectance
        // Transform from World space to Local space
        inDir = worldToLocalSpace(inDir);
        Pair<Vector3f, Spectrum> samplef = brdf.sampleF(convertVector3To3f(inDir.normalize()), convertVector3To3f(state.getNormal().normalize()));

        Vector3 refDir = localToWorldSpace(convertVector3fTo3(samplef.getKey()));

        Color d = spectrumToColor(brdf.f(convertVector3To3f(inDir).normalize(), convertVector3To3f(refDir).normalize(), "diffuse"));
        d = d.mul((float) Math.PI);
        Color lr = state.diffuse(d);

        // If no specular component exists
        if (brdf.f(convertVector3To3f(inDir), convertVector3To3f(refDir), "specular") == null) {
            return lr;
        }

        // Compute Fresnel term
        float cos = state.getCosND();
        cos = 1 - cos;
        float cos2 = cos * cos;
        float cos5 = cos2 * cos2 * cos;

        Color ret = Color.white();
        Color r = spectrumToColor(samplef.getValue());
        ret.sub(r);
        ret.mul(cos5);
        ret.add(r);

        Ray refRay = new Ray(state.getPoint(), refDir);
        return lr.add(ret.mul(state.traceReflection(refRay, 0)));
    }

    /**
     * Implements Forward Ray Tracing
     * Used for Global Illumination
     * @param state current state
     * @param power power of the incoming photon.
     */
    public void scatterPhoton(ShadingState state, Color power) {
        Color diffuse;
        state.faceforward();

        // make sure we are on the right side of the material
        if (Vector3.dot(state.getNormal(), state.getRay().getDirection()) > 0.0) {
            state.getNormal().negate();
            state.getGeoNormal().negate();
        }

        Vector3 inDir = state.getRay().getDirection();
        inDir = state.transformVectorWorldToObject(inDir);
        Pair<Vector3f, Spectrum> samplef = brdf.sampleF(convertVector3To3f(inDir.normalize()), convertVector3To3f(state.getNormal().normalize()));
        Vector3 refDir = state.transformVectorObjectToWorld(convertVector3fTo3(samplef.getKey()));

        Spectrum spectrumDiffuse = brdf.f(convertVector3To3f(inDir).normalize(), convertVector3To3f(refDir).normalize(), "diffuse");
        spectrumDiffuse.mul((float) Math.PI);
        diffuse = spectrumToColor(spectrumDiffuse);

        if(brdf.f(convertVector3To3f(inDir).normalize(), convertVector3To3f(refDir).normalize(), "specular") == null) {
            state.storePhoton(state.getRay().getDirection(), power, diffuse);
            float avg = diffuse.getAverage();
            double rnd = state.getRandom(0, 0, 1);

            if (rnd < avg) {
                // photon is scattered
                power.mul(diffuse).mul(1.0f / avg);
                OrthoNormalBasis onb = state.getBasis();
                double u = 2 * Math.PI * rnd / avg;
                double v = state.getRandom(0, 1, 1);
                float s = (float) Math.sqrt(v);
                float s1 = (float) Math.sqrt(1.0 - v);
                Vector3 w = new Vector3((float) Math.cos(u) * s, (float) Math.sin(u) * s, s1);
                w = onb.transform(w, new Vector3());
                state.traceDiffusePhoton(new Ray(state.getPoint(), w), power);
            }
        } else if(samplef.getValue() != null) {

            state.storePhoton(state.getRay().getDirection(), power, diffuse);
            float d = spectrumDiffuse.toScalar();

            float r = samplef.getValue().toScalar();

            // Russian roullette to determine unbiased path propagation if path will contribute
            double rnd = state.getRandom(0, 0, 1);
            if (rnd < d) {
                // photon is scattered
                power.mul(diffuse).mul(1.0f / d);
                OrthoNormalBasis onb = state.getBasis();
                double u = 2 * Math.PI * rnd / d;
                double v = state.getRandom(0, 1, 1);
                float s = (float) Math.sqrt(v);
                float s1 = (float) Math.sqrt(1.0 - v);
                Vector3 w = new Vector3((float) Math.cos(u) * s, (float) Math.sin(u) * s, s1);
                w = onb.transform(w, new Vector3());
                state.traceDiffusePhoton(new Ray(state.getPoint(), w), power);
            } else if (rnd < d + r) {
                float cos = -Vector3.dot(state.getNormal(), state.getRay().getDirection());
                power.mul(diffuse).mul(1.0f / d);
                // photon is reflected
                float dn = 2 * cos;
                Vector3 dir = new Vector3();
                dir.x = (dn * state.getNormal().x) + state.getRay().getDirection().x;
                dir.y = (dn * state.getNormal().y) + state.getRay().getDirection().y;
                dir.z = (dn * state.getNormal().z) + state.getRay().getDirection().z;
                state.traceReflectionPhoton(new Ray(state.getPoint(), dir), power);
            }
        }
    }

    private Vector3f convertVector3To3f(Vector3 vector) {
        Vector3f v = new Vector3f();
        v.x = vector.x;
        v.y = vector.y;
        v.z = vector.z;

        return v;
    }

    private Vector3 convertVector3fTo3(Vector3f vector) {
        Vector3 v = new Vector3();
        v.x = vector.x;
        v.y = vector.y;
        v.z = vector.z;

        return v;
    }

    private Color spectrumToColor(Spectrum spectrum) {
        return new Color(spectrum.getR(), spectrum.getG(), spectrum.getB());
    }

    private Vector3 worldToLocalSpace(Vector3 v) {
        return new Vector3(v);
    }

    private Vector3 localToWorldSpace(Vector3 v) {
        return new Vector3(v);
    }
}







