package libsvm

import (
	"container/list"
	"fmt"
)

type cacheNode struct {
	index    int           // column index for which this cacheNode is caching
	refCount int           // number of times this column was referenced
	data     []float64     // valid slice from cache::cacheBuffer if in the cache
	offset   int           // offset in cache::cacheBuffer data field is referring too
	element  *list.Element // list element (iterator) if in LRU list
}

type cache struct {
	head            []cacheNode // all the possible cached columns
	colSize         int         // size of each column
	cacheAvail      int         // number of additional columns we can store
	cacheBuffer     []float64   // pre-allocated buffer for cache
	availableOffset int         // next available offset in cacheBuffer
	hits, misses    int         // cache statistics
	cacheList       *list.List  // LRU list
}

const sizeOfFloat64 = 8

func (c *cache) getData(i int) ([]float64, bool) {
	var newData bool = true

	c.head[i].refCount++ // count reference to this index

	if c.head[i].offset != -1 {
		h := &(c.head[i])
		c.cacheList.Remove(h.element) // Remove from LRU list so we can re-insert into the back
		newData = false
		c.hits++
	}

	if newData {
		var useOffset int = 0
		// new data
		if c.cacheAvail == 0 { // no more space in cache
			// free a column

			//Remove the front of the LRU list (the least used)
			oldElement := c.cacheList.Front()
			value := c.cacheList.Remove(oldElement)
			old := value.(*cacheNode)

			useOffset = old.offset // reuse its memory
			old.offset = -1

			c.cacheAvail++
		} else {
			useOffset = c.availableOffset
			c.availableOffset += c.colSize
		}

		c.head[i].data = c.cacheBuffer[useOffset : useOffset+c.colSize]
		c.head[i].offset = useOffset // remember which offsef this cache line has been assigned

		c.cacheAvail--

		c.misses++
	}

	h := &(c.head[i])
	h.element = c.cacheList.PushBack(h) // Inserts it at the back of LRU circular list
	return c.head[i].data, newData
}

func (c cache) stats() {
	fmt.Printf("Cache misses:     %d\n", c.misses)
	fmt.Printf("Cache hits:       %d\n", c.hits)
	fmt.Printf("Cache efficiency: %.6f%%\n", float32(c.hits)/float32(c.hits+c.misses)*100)
}

func computeCacheSize(colSize int) int {
	var cacheSize int = 500 // MB (default)

	cacheSizeBytes := cacheSize * (1 << 20)
	numCols := cacheSizeBytes / (colSize * sizeOfFloat64) // num of columns we can store
	numCols = maxi(2, numCols)                            // we should be able to store at least 2

	return numCols
}

func NewCache(l, colSize int) *cache {

	colCacheSize := computeCacheSize(colSize) // number of columns we can cache

	head := make([]cacheNode, l)
	for i := 0; i < l; i++ {
		head[i].index = i
		head[i].refCount = 0
		head[i].data = nil
		head[i].offset = -1
	}

	c := cache{head: head, colSize: colSize, cacheAvail: colCacheSize, availableOffset: 0, hits: 0, misses: 0}
	c.cacheBuffer = make([]float64, colCacheSize*colSize)
	c.cacheList = list.New()

	return &c
}
