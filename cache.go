/*
** Copyright 2014 Edward Walker
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
** http ://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
**
** Description: Caches Q matrix rows.  The cache implements a LRU (Last Recently Used) eviction policy.
** @author: Ed Walker
 */
package libSvm

import (
	"container/list"
	"fmt"
)

// To allow the LRU cache to store values of a different type, modify BOTH cacheDataType and sizeOfCacheDataType
type cacheDataType float32

const sizeOfCacheDataType = 4

type cacheNode struct {
	index    int             // row index for which this cacheNode is caching
	refCount int             // number of times this row was referenced
	data     []cacheDataType // valid slice from cache::cacheBuffer if in the cache
	offset   int             // offset in cache::cacheBuffer data field is referring too
	element  *list.Element   // list element (iterator) if in LRU list
}

type cache struct {
	head            []cacheNode     // all the possible cached rows
	rowSize         int             // size of each row
	cacheAvail      int             // number of additional rows we can store
	cacheBuffer     []cacheDataType // pre-allocated buffer for cache
	availableOffset int             // next available offset in cacheBuffer
	hits, misses    int             // cache statistics
	cacheList       *list.List      // LRU list
}

func (c *cache) getData(i int) ([]cacheDataType, bool) {
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
			// free a row

			//Remove the front of the LRU list (the last used)
			oldElement := c.cacheList.Front()
			value := c.cacheList.Remove(oldElement)
			old := value.(*cacheNode)

			useOffset = old.offset // reuse its memory
			old.offset = -1

			c.cacheAvail++
		} else {
			useOffset = c.availableOffset
			c.availableOffset += c.rowSize
		}

		c.head[i].data = c.cacheBuffer[useOffset : useOffset+c.rowSize]
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

func computeCacheSize(rowSize, cacheSize int) int {

	cacheSizeBytes := cacheSize * (1 << 20)                     // multiply by 1 Mbytes
	numCols := cacheSizeBytes / (rowSize * sizeOfCacheDataType) // num of rows we can store
	numCols = maxi(2, numCols)                                  // we should be able to store at least 2

	return numCols
}

func newCache(l, rowSize, cacheSize int) *cache {

	rowCacheSize := computeCacheSize(rowSize, cacheSize) // number of rows we can cache

	head := make([]cacheNode, l)
	for i := 0; i < l; i++ {
		head[i].index = i
		head[i].refCount = 0
		head[i].data = nil
		head[i].offset = -1
	}

	c := cache{head: head, rowSize: rowSize, cacheAvail: rowCacheSize, availableOffset: 0, hits: 0, misses: 0}
	c.cacheBuffer = make([]cacheDataType, rowCacheSize*rowSize)
	c.cacheList = list.New()

	return &c
}
