#include <map>

#include <vigra/multi_array.hxx>

vigra::NumpyAnyArray candidateSegToRagSeg(
    vigra::NumpyArray<2, uint32_t> ragLabels,
    vigra::NumpyArray<2, uint32_t> candidateLabels,
    vigra::NumpyArray<1, vigra::TinyVector<uint64_t, 2> > uvIds,
    vigra::NumpyArray<1, float> out
){  
    out.reshapeIfEmpty(uvIds.shape());

    std::map<
        uint32_t,
        std::map<uint32_t, uint32_t>
    > overlapsWith;
    std::map<uint32_t, uint32_t> counters;

    //std::cout<<"ol mat\n";
    for(auto i=0; i<ragLabels.size(); ++i){
        const auto rl = ragLabels[i];
        const auto cl = candidateLabels[i]; 
        ++overlapsWith[rl][cl];
        ++counters[rl];
    }
    //std::cout<<"feat\n";
    for(auto e=0; e<uvIds.size(); ++e){
        const auto u = uvIds[e][0];
        const auto v = uvIds[e][1];
        const auto sU = float(counters[u]);
        const auto sV = float(counters[v]);
        const auto & olU = overlapsWith[u];
        const auto & olV = overlapsWith[v];

        auto isDiff = 0.0;
        auto cc = 0.0;
        for(const auto & keyAndSizeU : olU)
        for(const auto & keyAndSizeV : olV){

            auto keyU =  keyAndSizeU.first;
            auto rSizeU = float(keyAndSizeU.second)/sU;
            auto keyV =  keyAndSizeV.first;
            auto rSizeV = float(keyAndSizeV.second)/sV;

            if(keyU != keyV){
                isDiff += (rSizeU * rSizeV);
            }
        }
        out[e] = isDiff;
    }
    //std::cout<<"done\n";
    return out;
}
