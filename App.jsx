import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";
const USER_ID = 1;

const cleanTitle = (title = "") => title.replace(/\s*\(\d{4}\)\s*$/, "");
const splitGenres = (value = "") => value.split("|").filter(Boolean);
const yearOf = (movie) => movie?.year || movie?.title?.match(/\((\d{4})\)/)?.[1] || "";

function Icon({ name, size = 18 }) {
  const p = {
    search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
    play: <path d="m9 7 8 5-8 5V7Z" fill="currentColor" stroke="none"/>,
    plus: <><path d="M12 5v14"/><path d="M5 12h14"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    brain: <><path d="M9.5 4a3 3 0 0 0-3 3v.5a3 3 0 0 0-2 5.1 3 3 0 0 0 2.5 4.9h1v1.5"/><path d="M14.5 4a3 3 0 0 1 3 3v.5a3 3 0 0 1 2 5.1 3 3 0 0 1-2.5 4.9h-1v1.5"/><path d="M12 4v16"/><path d="M8 8h2M14 8h2M8 13h2M14 13h2"/></>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 10v6M12 7h.01"/></>,
    x: <><path d="m6 6 12 12"/><path d="m18 6-12 12"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    spark: <path d="m12 3 1.6 5.4L19 10l-5.4 1.6L12 17l-1.6-5.4L5 10l5.4-1.6L12 3Z"/>,
    refresh: <><path d="M20 11a8 8 0 0 0-14-4L4 9"/><path d="M4 4v5h5"/><path d="M4 13a8 8 0 0 0 14 4l2-2"/><path d="M20 20v-5h-5"/></>
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{p[name]}</svg>;
}

function Poster({ movie, className = "" }) {
  return movie?.poster_url
    ? <img className={`poster ${className}`} src={movie.poster_url} alt={movie.title} />
    : <div className={`poster poster-empty ${className}`}><span>🎬</span><small>No artwork</small></div>;
}

function Rating({ value }) {
  return value == null ? null : <span className="rating">★ {Number(value).toFixed(1)}</span>;
}

function MovieCard({ movie, index, onOpen, saved, onSave }) {
  const match = Math.round((movie.content_score || 0) * 100);
  return (
    <article className="movie-card">
      <button className="poster-button" onClick={() => onOpen(movie)}>
        <Poster movie={movie} />
        <span className="rank">#{String(index + 1).padStart(2, "0")}</span>
        <span className="match">{match}% content match</span>
        <div className="hover">
          <span className="play-circle"><Icon name="play" size={19}/></span>
          <span>See why</span>
        </div>
      </button>
      <div className="card-body">
        <button className="movie-title" onClick={() => onOpen(movie)}>{cleanTitle(movie.title)}</button>
        <div className="meta"><span>{yearOf(movie)}</span><Rating value={movie.avg_rating}/></div>
        <div className="genres">{splitGenres(movie.genres).slice(0,2).join(" • ")}</div>
        <button className={`save ${saved ? "saved" : ""}`} onClick={() => onSave(movie)}>
          {saved ? <Icon name="check" size={13}/> : <Icon name="plus" size={13}/>}
        </button>
      </div>
    </article>
  );
}

function WhyYoullLike({ movie }) {
  if (!movie) return null;
  const e = movie.explanation || {};
  const content = Math.round((e.content_match?.score ?? movie.content_score ?? 0) * 100);
  const quality = Math.round((e.quality?.score ?? 0) * 100);
  const prediction = Number(e.predicted_rating ?? movie.predicted_rating ?? 0).toFixed(2);
  return (
    <div className="why-box">
      <div className="why-title">
        <div className="brain"><Icon name="brain" size={19}/></div>
        <div><span className="label">MODEL EXPLANATION</span><h3>Why you'll like this</h3></div>
      </div>
      <p>{e.summary || movie.reason || "This movie scored strongly across the recommendation signals."}</p>
      <div className="metrics">
        <div><span>CONTENT MATCH</span><b>{content}%</b><div className="bar"><i style={{width:`${content}%`}}/></div><small>TF-IDF similarity</small></div>
        <div><span>PREDICTED RATING</span><b>{prediction}/5</b><div className="bar"><i style={{width:`${Number(prediction)/5*100}%`}}/></div><small>Personalized estimate</small></div>
        <div><span>QUALITY SIGNAL</span><b>{quality}%</b><div className="bar"><i style={{width:`${quality}%`}}/></div><small>{e.quality?.rating_count ?? movie.rating_count ?? 0} ratings</small></div>
      </div>
      {e.why_youll_like_it?.length > 0 && <div className="reasons">{e.why_youll_like_it.map((x,i)=><div key={i}><span>✓</span>{x}</div>)}</div>}
      {e.based_on_liked_movies?.length > 0 && <div className="based"><span className="label">LEARNED FROM YOUR HIGH RATINGS</span><div>{e.based_on_liked_movies.map(x=><span key={x}>{cleanTitle(x)}</span>)}</div></div>}
    </div>
  );
}

function Details({ movie, onClose, saved, onSave }) {
  if (!movie) return null;
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="details" onClick={e=>e.stopPropagation()}>
        <button className="close" onClick={onClose}><Icon name="x"/></button>
        <div className="detail-hero" style={movie.backdrop_url ? {backgroundImage:`url(${movie.backdrop_url})`} : {}}>
          <div className="detail-shade"/>
          <div className="detail-heading">
            <Poster movie={movie} className="detail-poster"/>
            <div>
              <span className="label red">PERSONALIZED RECOMMENDATION</span>
              <h2>{movie.title}</h2>
              <div className="detail-meta"><span>{yearOf(movie)}</span><span>{movie.runtime ? `${movie.runtime} min` : "—"}</span><Rating value={movie.avg_rating}/>{movie.tmdb_rating && <span>TMDB {Number(movie.tmdb_rating).toFixed(1)}</span>}</div>
              <div className="detail-genres">{splitGenres(movie.genres).join(" • ")}</div>
            </div>
          </div>
        </div>
        <div className="detail-content">
          <div className="detail-actions">
            <button className="watch"><Icon name="play" size={15}/> Explore movie</button>
            <button className={`list ${saved ? "saved" : ""}`} onClick={()=>onSave(movie)}>{saved ? <Icon name="check" size={15}/> : <Icon name="plus" size={15}/>} {saved ? "In My List" : "Add to My List"}</button>
          </div>
          <p className="overview">{movie.overview || "No synopsis is available for this title."}</p>
          <div className="credits">
            {movie.director && <div><span>DIRECTOR</span><b>{movie.director}</b></div>}
            {movie.cast && <div><span>CAST</span><b>{movie.cast.split(" ").slice(0,18).join(" ")}</b></div>}
          </div>
          <WhyYoullLike movie={movie}/>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [recommendationPool, setRecommendationPool] = useState([]);
  const [refreshIndex, setRefreshIndex] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);
  const [recommending, setRecommending] = useState(false);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("personalized");
  const [seed, setSeed] = useState("");
  const [favorites, setFavorites] = useState(()=>JSON.parse(localStorage.getItem("movie-favorites") || "[]"));
  const savedIds = useMemo(()=>new Set(favorites.map(x=>x.movieId)),[favorites]);

  const applyPool = (pool, index = 0) => {
    const size = 20;
    if (!pool.length) { setRecommendations([]); return; }
    const start = (index * 11) % pool.length;
    const rotated = [...pool.slice(start), ...pool.slice(0, start)];
    const page = rotated.slice(0, Math.min(size, pool.length));
    setRecommendations(page);
  };

  const personalized = async () => {
    setRecommending(true); setError("");
    try {
      const r=await fetch(`${API}/recommend/user/${USER_ID}?limit=50`);
      const d=await r.json(); if(!r.ok) throw new Error(d.detail || "Could not load recommendations");
      const pool=d.recommendations||[];
      setRecommendationPool(pool); setRefreshIndex(0); applyPool(pool, 0);
      setMode("personalized"); setSeed("");
    } catch(e){setError(e.message)} finally{setRecommending(false);setLoading(false)}
  };

  const refreshRecommendations = async () => {
    if (recommending || refreshing) return;
    setRefreshing(true); setError("");
    try {
      let pool = recommendationPool;
      if (!pool.length) {
        const endpoint = mode === "movie"
          ? `${API}/recommend/${encodeURIComponent(seed)}?user_id=${USER_ID}&limit=50`
          : `${API}/recommend/user/${USER_ID}?limit=50`;
        const r = await fetch(endpoint);
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || "Could not refresh recommendations");
        pool = d.recommendations || [];
        setRecommendationPool(pool);
      }
      const next = refreshIndex + 1;
      setRefreshIndex(next);
      applyPool(pool, next);
    } catch(e) { setError(e.message); }
    finally { setRefreshing(false); setLoading(false); }
  };

  const recommendFrom = async (title) => {
    if(!title.trim()) return;
    setRecommending(true); setError(""); setSearchResults([]);
    try {
      const r=await fetch(`${API}/recommend/${encodeURIComponent(title.trim())}?user_id=${USER_ID}&limit=50`);
      const d=await r.json(); if(!r.ok) throw new Error(d.detail || "Movie not found");
      const pool=d.recommendations||[];
      setRecommendationPool(pool); setRefreshIndex(0); applyPool(pool, 0);
      setMode("movie"); setSeed(title.trim()); setQuery(title.trim());
    } catch(e){setError(e.message)} finally{setRecommending(false);setLoading(false)}
  };

  useEffect(()=>{ personalized(); },[]);

  useEffect(()=>{
    const t=setTimeout(async()=>{
      if(query.trim().length<2){setSearchResults([]);return}
      setSearching(true);
      try{
        const r=await fetch(`${API}/search?query=${encodeURIComponent(query)}&limit=7`);
        const d=await r.json(); setSearchResults(d.results||[]);
      }catch{setSearchResults([])}finally{setSearching(false)}
    },220);
    return()=>clearTimeout(t);
  },[query]);

  const save = movie => {
    const next=savedIds.has(movie.movieId)?favorites.filter(x=>x.movieId!==movie.movieId):[...favorites,movie];
    setFavorites(next); localStorage.setItem("movie-favorites",JSON.stringify(next));
  };

  const chooseSearchResult = movie => recommendFrom(movie.title);
  const hero=recommendations[0];
  const content=[...recommendations].sort((a,b)=>(b.content_score||0)-(a.content_score||0));
  const predicted=[...recommendations].sort((a,b)=>(b.predicted_rating||0)-(a.predicted_rating||0));

  return (
    <div className="app">
      <header className="nav">
        <div className="brand"><span className="mark">M</span><span>MOVIE<span>ML</span></span></div>
        <nav><button className="active">Recommendations</button><button onClick={personalized}>For You</button><button onClick={()=>document.getElementById("model")?.scrollIntoView({behavior:"smooth"})}>How it works</button></nav>
        <div className="status"><i/> ML engine online <span className="user">USER {USER_ID}</span></div>
      </header>

      <main>
        <section className="hero" style={hero?.backdrop_url?{backgroundImage:`url(${hero.backdrop_url})`}:{}}>
          <div className="hero-bg"/>
          <div className="hero-content">
            <div className="eyebrow"><span/> {mode==="personalized"?"PERSONALIZED FOR YOU":"RECOMMENDATIONS BASED ON YOUR SEARCH"}</div>
            <h1>{hero ? cleanTitle(hero.title) : "Discover your next favorite."}</h1>
            <div className="hero-meta">{hero&&<><span>{yearOf(hero)}</span><span>{hero.runtime?`${hero.runtime} min`:"—"}</span><Rating value={hero.avg_rating}/><span>{splitGenres(hero.genres).slice(0,3).join(" • ")}</span></>}</div>
            <p>{hero?.overview || "Your recommendation model combines content similarity, collaborative filtering and quality-aware ranking."}</p>
            <div className="hero-actions">
              {hero&&<button className="primary" onClick={()=>setSelected(hero)}><Icon name="info" size={15}/> Why this movie</button>}
              <button className="secondary" onClick={refreshRecommendations} disabled={recommending || refreshing}><Icon name="refresh" size={15}/>{refreshing?"Refreshing...":"Refresh recommendations"}</button>
            </div>
            {hero&&<div className="hero-model"><Icon name="spark" size={14}/><span>Predicted for you <b>{Number(hero.predicted_rating||0).toFixed(2)}/5</b></span><em/><span>Hybrid score <b>{Number(hero.hybrid_score||0).toFixed(3)}</b></span></div>}
          </div>
        </section>

        <section className="search-area">
          <div className="search-label"><span className="label">DISCOVER BY MOVIE</span><b>Search a movie — we'll re-rank recommendations around it</b></div>
          <div className="search-box">
            <Icon name="search" size={19}/>
            <input value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>e.key==="Enter" && recommendFrom(query)} placeholder="Try The Dark Knight, Inception, Pulp Fiction..." />
            {searching&&<span className="spinner"/>}
            {query&&<button className="clear" onClick={()=>{setQuery("");setSearchResults([])}}><Icon name="x" size={15}/></button>}
            <button className="recommend-button" onClick={()=>recommendFrom(query)} disabled={!query.trim() || recommending}>Recommend</button>
          </div>
          {searchResults.length>0&&<div className="search-results">
            <div className="search-result-head">SELECT A MOVIE <span>{searchResults.length} matches</span></div>
            {searchResults.map(m=><button key={m.movieId} onClick={()=>chooseSearchResult(m)}>
              <Poster movie={m}/><span className="result-copy"><b>{m.title}</b><small>{yearOf(m)} • {splitGenres(m.genres).slice(0,3).join(" • ")}</small></span><span className="result-cta">Recommend <Icon name="chevron" size={14}/></span>
            </button>)}
          </div>}
          <div className="quick-searches"><span>Try:</span>{["The Dark Knight","Inception","Pulp Fiction","The Godfather"].map(x=><button key={x} onClick={()=>recommendFrom(x)}>{x}</button>)}</div>
        </section>

        {error&&<div className="error">{error}</div>}

        <section className="results" id="recommendations">
          <div className="results-top">
            <div><span className="label">{mode==="personalized"?"YOUR LEARNED PREFERENCE PROFILE":"RE-RANKED FROM SEED MOVIE"}</span><h2>{mode==="personalized"?"Recommended For You":`Because you searched “${cleanTitle(seed)}”`}</h2><p>{mode==="personalized"?"Ranked from your rating history using the hybrid recommendation model.":"The seed movie is used to discover similar content, then your user preferences and quality signals influence the ranking."}</p></div>
            <div className="results-tools">
              <div className="result-count">{recommending?"RUNNING MODEL":`${recommendations.length} SHOWN · ${recommendationPool.length || recommendations.length} CANDIDATES`}</div>
              {!recommending && recommendations.length > 0 && <button className="refresh-feed" onClick={refreshRecommendations} disabled={refreshing}><Icon name="refresh" size={14}/>{refreshing ? "Refreshing" : "New recommendations"}</button>}
            </div>
          </div>

          {recommending ? <div className="loading-row">{[1,2,3,4,5].map(x=><div className="sk" key={x}/>)}</div> :
            recommendations.length ? <>
              <div className="section-title"><h3>Best overall matches</h3><span>Hybrid ranking</span></div>
              <div className="grid">{recommendations.slice(0,10).map((m,i)=><MovieCard key={m.movieId} movie={m} index={i} onOpen={setSelected} saved={savedIds.has(m.movieId)} onSave={save}/>)}</div>
              <div className="section-title second"><h3>Highest predicted for you</h3><span>Personalized rating signal</span></div>
              <div className="grid">{predicted.slice(0,8).map((m,i)=><MovieCard key={`p-${m.movieId}`} movie={m} index={i} onOpen={setSelected} saved={savedIds.has(m.movieId)} onSave={save}/>)}</div>
              <div className="section-title second"><h3>Closest content matches</h3><span>TF-IDF similarity</span></div>
              <div className="grid">{content.slice(0,8).map((m,i)=><MovieCard key={`c-${m.movieId}`} movie={m} index={i} onOpen={setSelected} saved={savedIds.has(m.movieId)} onSave={save}/>)}</div>
            </> : <div className="empty"><Icon name="brain" size={30}/><h3>Your recommendation feed is ready</h3><p>Search for a movie above or refresh your personalized recommendations.</p></div>}
        </section>

        <section className="model-section" id="model">
          <div><span className="label">THE ML BEHIND THE UI</span><h2>Three signals. One personalized ranking.</h2><p>The interface exposes the same signals returned by your FastAPI model, so the recommendation isn't a black box.</p></div>
          <div className="pipeline">
            <div><span>01</span><h3>Collaborative filtering</h3><p>Latent factors learn patterns from user–movie ratings.</p><b>60%</b></div>
            <div><span>02</span><h3>Content similarity</h3><p>TF-IDF compares title, genres, overview, keywords, cast and director.</p><b>25%</b></div>
            <div><span>03</span><h3>Quality signal</h3><p>Bayesian quality helps prevent weakly-rated titles from dominating.</p><b>15%</b></div>
          </div>
        </section>
      </main>

      <footer><div className="brand"><span className="mark small">M</span><span>MOVIE<span>ML</span></span></div><p>Personalized movie discovery • Hybrid recommendation engine</p></footer>

      {selected&&<Details movie={selected} onClose={()=>setSelected(null)} saved={savedIds.has(selected.movieId)} onSave={save}/>}
    </div>
  );
}
export default App;
