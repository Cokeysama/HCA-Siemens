from DistMeasures import jaccard_distance_2
import heapq
import math


class ClusterNode(object):
    def __init__(self, element=None, children1=None, children2=None):
        if children1 and children2:
            self.elements = children1.elements | children2.elements
            self.children = {children1, children2}
            self.repr = children1.repr if children1.repr > children2.repr else children2.repr
        elif element is not None:
            self.elements = {element}
            self.repr = element

        self.size = len(self.elements)

    def __str__(self):
        return str(self.repr)


class NodePair:
    def __init__(self, node1, node2, dist):
        self.node1 = node1
        self.node2 = node2
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other.dist


class SLINK(object):

    def __init__(self, samples=None, metric=jaccard_distance_2):

        self.samples = samples
        self.metric = metric
        self.clusters = [ClusterNode(element=i) for i in range(1, len(samples) + 1)]
        self.n_clusters = len(samples)

        self.__pointer_repr()
        self.__creat_distheapq()

    def __pointer_repr(self):
        samples = self.samples
        λ, π, M = [], [], []
        for n in range(self.n_clusters):
            λ.append(math.inf)
            π.append(n + 1)
            M.append(None)
            for i in range(n):
                M[i] = self.metric(samples[i], samples[n])
            for i in range(n):
                if λ[i] >= M[i]:
                    M[π[i] - 1] = min(M[π[i] - 1], λ[i])
                    λ[i], π[i] = M[i], n + 1
                else:
                    M[π[i] - 1] = min(M[π[i] - 1], M[i])
            for i in range(n):
                if λ[i] >= λ[π[i] - 1]:
                    π[i] = n + 1
        self.λ, self.π = λ, π

    def __creat_distheapq(self):
        λ, π = self.λ, self.π
        n_samples = len(self.samples)
        nplist = []
        for i in range(n_samples):
            nplist.append(NodePair(i + 1, π[i], λ[i]))
        heapq.heapify(nplist)
        self.distheapq = nplist

    def __merge(self):
        clusters = self.clusters
        mergenode = heapq.heappop(self.distheapq)
        node1, node2 = mergenode.node1, mergenode.node2
        cluster1, cluster2 = None, None

        for cluster in clusters[::-1]:
            if cluster.repr == node1:
                cluster1 = cluster
                clusters.remove(cluster)
            elif cluster.repr == node2:
                cluster2 = cluster
                clusters.remove(cluster)

        newnode = ClusterNode(children1=cluster1, children2=cluster2)
        clusters.append(newnode)


    def clustering(self, times = None, target = None):

        size = len(self.clusters)

        if target:
            while size > target:
                self.__merge()
        elif times:
            for i in range(times):
                if size > 1:
                    self.__merge()
                else:
                    break
        return self.clusters