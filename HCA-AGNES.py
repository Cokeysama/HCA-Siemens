import time
from DistMeasures import jaccard_distance

class ClusterNode(object):
    def __init__(self, samples = None, children1 = None, children2 = None):
        if children1 and children2:
            self.samples = children1.samples|children2.samples
            self.children = {children1, children2}
        elif samples is not None:
            self.samples = {samples}
        self.n_samples = len(self.samples)

class HierarchicalClustering(object):

    def __init__(self, samples = None, target = 1, linkage = 'avglkge', metric = jaccard_distance):

        self.samples = samples
        self.linkage = linkage
        self.metric = metric
        self.target = target

        self.distsort = None
        self.distMatrix = None
        self.clusters = [ClusterNode(i) for i in range(len(samples))]
        self.n_clusters = len(samples)

        self.__creat_distMatrix()
        self.__creat_distsort()

    def avglkge(self,n1,n2):
        TD = 0.0
        n = 0
        for i in n1.samples:
            for j in n2.samples:
                n += 1
                TD += self.distMatrix[i][j]
        return (TD / n)

    def __creat_distMatrix(self):
        if self.distMatrix is None:
            n_samples = len(self.samples)
            distMatrix = [[0]*n_samples for i in range(n_samples)]
            for i in range(n_samples-1):
                for j in range(i + 1, n_samples):
                    dist = round(self.metric(self.samples[i], self.samples[j]),3)
                    distMatrix[i][j], distMatrix[j][i] = dist, dist
            self.distMatrix = distMatrix

    def __creat_distsort(self):
        if self.distsort is None:
            n_samples = len(self.samples)
            distMatrix = self.distMatrix
            distlist = []
            for i in range(n_samples-1):
                for j in range(i + 1, n_samples):
                    dist = distMatrix[i][j]
                    distlist.append(({i},{j},dist))
            self.distsort = sorted(distlist, key=lambda d:d[2],reverse = True)

    def __merge_mix(self):
        cluster1, cluster2 = self.distsort[-1][0], self.distsort[-1][1]
        merge_node1,merge_node2 = None,None
        i = 0
        while merge_node1 is None or merge_node2 is None:

            if self.clusters[i].samples == cluster1:
                merge_node1 = self.clusters[i]
                self.clusters.pop(i)
            elif self.clusters[i].samples == cluster2:
                merge_node2 = self.clusters[i]
                self.clusters.pop(i)
            else: i += 1

        new_CL = ClusterNode(children1=merge_node1, children2=merge_node2)
        self.clusters.append(new_CL)

    def __update_distsort(self):
        distsort = self.distsort
        new_CL = self.clusters[-1]
        del_id1, del_id2 = distsort[-1][0], distsort[-1][1]
        for i in range(len(distsort)-1,-1,-1):
            if del_id1 == distsort[i][0] or del_id2 == distsort[i][0] or del_id1 == distsort[i][1] or del_id2 == distsort[i][1]:
                distsort.pop(i)
                
        for i in self.clusters[:-1]:
            dist = self.avglkge(i, new_CL)
            new_sortitem = (i.samples, new_CL.samples, dist)
            left, right = 0, len(distsort) - 1
            while left <= right:
                mid = ((right - left) >> 1) + left
                if distsort[mid][2] == dist:
                    distsort.insert(mid, new_sortitem)
                    break
                elif distsort[mid][2] < dist:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                distsort.insert(right - 1, new_sortitem)



    def clustering(self):

        while self.n_clusters > self.target:
            self.__merge_mix()
            self.__update_distsort()
            self.n_clusters = len(self.clusters)
            for i in self.clusters:
                print(i.samples, end='')
            print()
def main():
    samples = [[1, 0, 1, 1, 0, 0, 0, 0, 1],
               [1, 0, 0, 1, 0, 1, 1, 0, 1],
               [0, 1, 1, 0, 0, 0, 1, 1, 0],
               [0, 1, 0, 0, 1, 1, 0, 1, 0],
               [1, 0, 0, 1, 0, 1, 1, 1, 0],
               [0, 1, 1, 0, 0, 0, 1, 0, 1],]
    fit = HierarchicalClustering(samples)
    fit.clustering()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('程序执行时间: ', end - start)
