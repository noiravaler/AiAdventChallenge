package orbit

import (
	"sync"
	"time"
)

// Cache — простой TTL-кэш для ответов провайдеров (геокодинг, погода, Луна).
// Значения считаются неизменяемыми после записи.
type Cache struct {
	mu    sync.Mutex
	items map[string]cacheItem
}

type cacheItem struct {
	value   any
	expires time.Time
}

func NewCache() *Cache {
	return &Cache{items: map[string]cacheItem{}}
}

// Get возвращает значение, если оно живо.
func (c *Cache) Get(key string) (any, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	it, ok := c.items[key]
	if !ok || time.Now().After(it.expires) {
		if ok {
			delete(c.items, key)
		}
		return nil, false
	}
	return it.value, true
}

// Set кладёт значение с TTL.
func (c *Cache) Set(key string, value any, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items[key] = cacheItem{value: value, expires: time.Now().Add(ttl)}
}

// TTL провайдеров (время жизни записей кэша).
const (
	ttlGeo     = 72 * time.Hour // города не переезжают
	ttlWeather = 30 * time.Minute
	ttlMoon    = 12 * time.Hour
)
